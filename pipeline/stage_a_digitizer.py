#!/usr/bin/env python3
"""
Stage A: CRF Digitizer using Qwen2.5-VL-7B
Extracts questions, options, and bounding boxes from CRF pages
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import io
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Normalized bounding box coordinates"""
    x1: float  # [0, 1]
    y1: float  # [0, 1]
    x2: float  # [0, 1]
    y2: float  # [0, 1]
    
    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    @classmethod
    def from_pdf_rect(cls, rect: fitz.Rect, page_width: float, page_height: float) -> 'BoundingBox':
        """Convert PyMuPDF rect to normalized bbox"""
        return cls(
            x1=rect.x0 / page_width,
            y1=rect.y0 / page_height,
            x2=rect.x1 / page_width,
            y2=rect.y1 / page_height
        )


@dataclass
class ExtractedLine:
    """Text line with stable ID and bbox"""
    line_id: str
    text: str
    bbox: BoundingBox
    page_num: int


@dataclass
class CRFItem:
    """Extracted CRF item (question or form field)"""
    item_id: str
    type: str  # "question", "field", "checkbox", "radio", "info"
    text: str
    bbox_xyxy: List[float]
    bbox_ref: str  # Reference to line_id
    options: List[Dict] = None
    form: Optional[str] = None  # Form name (e.g., "Baseline", "Follow-up")
    section: Optional[str] = None  # Section within the form (e.g., "Inclusion Criteria")
    subsection: Optional[str] = None  # Subsection if present
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        if not d['options']:
            d.pop('options')
        if not d['form']:
            d.pop('form')
        if not d['section']:
            d.pop('section')
        if not d['subsection']:
            d.pop('subsection')
        return d


class CRFDigitizer:
    """Stage A: Digitize CRF pages using Qwen2.5-VL-7B"""
    
    # Note: SYSTEM_MSG is not used in the current implementation 
    # The prompts are defined directly in each extraction method
    SYSTEM_MSG = None
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", 
                 device: str = "auto", use_4bit: bool = True, fallback_only: bool = False,
                 dpi: int = 300, clear_cache_per_page: bool = True, use_three_pass: bool = True):
        """
        Initialize digitizer with Qwen-VL model
        
        Args:
            model_path: HuggingFace model path or local path
            device: Device to use (auto, cuda, cpu)
            use_4bit: Use 4-bit quantization for memory efficiency
            fallback_only: Use only fallback extraction without vision model
            dpi: DPI for rendering PDF pages (default: 300)
            clear_cache_per_page: Clear GPU cache between pages
            use_three_pass: Use three-pass extraction (always True for new implementation)
        """
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_4bit = use_4bit
        self.fallback_only = fallback_only
        self.dpi = dpi
        self.clear_cache_per_page = clear_cache_per_page
        self.use_three_pass = use_three_pass
        
        # Initialize model and processor
        if not fallback_only:
            self._init_model()
        else:
            logger.info("Using fallback-only mode (no vision model)")
        
        # Output directories
        self.output_dir = Path("crf_json")
        self.output_dir.mkdir(exist_ok=True)
        
        # Track sequence number across pages
        self.current_seq = 1
        # Track question number across pages
        self.global_question_counter = 0
        
    def _init_model(self):
        """Initialize Qwen-VL model with proper configuration"""
        logger.info(f"Loading Qwen2.5-VL-7B model...")
        
        # Model configuration
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "trust_remote_code": True,
        }
        
        # Use device_map for CUDA
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        if self.use_4bit and self.device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load model - use the approach from reference code
        if Qwen2_5_VLForConditionalGeneration is not None:
            # Use specific Qwen2.5 class if available
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            logger.info("Loaded model with Qwen2_5_VLForConditionalGeneration")
        else:
            # Use AutoModelForVision2Seq as fallback
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            logger.info("Loaded model with AutoModelForVision2Seq")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Don't try to move the model if device_map is used (model is already distributed)
        # Only move to device if we're not using device_map
        if "device_map" not in model_kwargs and self.device != "auto":
            self.model = self.model.to(self.device)
            
        logger.info("Model loaded successfully")
        
    def extract_page_primitives(self, pdf_path: Path, page_num: int) -> Tuple[Image.Image, List[ExtractedLine]]:
        """
        Extract page image and text lines with bounding boxes
        
        Returns:
            (page_image, extracted_lines)
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Get page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # Render page to image (high resolution for better OCR)
        mat = fitz.Matrix(self.dpi/72, self.dpi/72)  # Use configured DPI
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        page_image = Image.open(io.BytesIO(img_data))
        
        # Extract text blocks with coordinates
        blocks = page.get_text("blocks")
        lines = []
        
        for block_idx, block in enumerate(blocks):
            if block[6] == 0:  # Text block (not image)
                # Split block into lines
                block_text = block[4]
                block_rect = fitz.Rect(block[:4])
                
                # Create line entry
                line = ExtractedLine(
                    line_id=f"line_{page_num:03d}_{block_idx:03d}",
                    text=block_text.strip(),
                    bbox=BoundingBox.from_pdf_rect(block_rect, page_width, page_height),
                    page_num=page_num
                )
                lines.append(line)
        
        doc.close()
        return page_image, lines
        
        
    def digitize_page(self, page_image: Image.Image, lines: List[ExtractedLine], 
                     page_num: int, current_form: Optional[str] = None, current_section: Optional[str] = None, 
                     max_retries: int = 3) -> Tuple[List[Dict], Optional[str], Optional[str]]:
        """
        Use Qwen-VL to extract CRF items from page using 3 separate queries
        
        Args:
            page_image: PIL Image of the page
            lines: Extracted text lines
            page_num: Page number
            current_form: Current form name from previous pages
            current_section: Current section from previous pages
            max_retries: Number of retries for extraction
        
        Returns:
            Tuple of (List of extracted items as dicts, updated form name, updated section)
        """
        # Use fallback if in fallback-only mode
        if self.fallback_only:
            items = self._fallback_extraction(lines, page_num, current_section)
            return items, current_form, current_section
        
        # Clear memory cache for new page session
        if self.clear_cache_per_page and self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Step 1: Extract question-input pairs
        logger.info(f"[Page {page_num + 1}] Step 1/3: Extracting question-input pairs...")
        question_items = self._extract_questions(page_image, page_num)
        
        # Step 2: Extract form name first
        logger.info(f"[Page {page_num + 1}] Step 2/3: Extracting form name...")
        form_items = self._extract_form_name(page_image, page_num)
        
        # Step 3: Extract section headers (pass questions and form items to avoid duplicates)
        logger.info(f"[Page {page_num + 1}] Step 3/3: Extracting section headers...")
        section_items = self._extract_sections(page_image, page_num, question_items, form_items)
        
        # Clean sections that are too similar to questions
        logger.info(f"[Page {page_num + 1}] Cleaning sections similar to questions...")
        section_items = self._clean_similar_sections(section_items, question_items)
        
        # Process sections and assign to questions intelligently
        all_items = []
        
        # Add form items first
        for item in form_items:
            if item.get('text'):
                current_form = item['text']
                all_items.append(item)
        
        # Create a map of question positions for section assignment
        question_positions = {}
        for idx, item in enumerate(question_items):
            if item.get('tag') == '<Q>' and 'qid' in item:
                question_positions[item['qid']] = idx
        
        # Process sections - initial assignment will be done by _assign_sections_by_question_below
        # Just add section headers to all_items for now
        if section_items:
            all_items.extend(section_items)
        
        # Update current_section to the last section found
        if section_items:
            last_section = section_items[-1]
            if last_section.get('text'):
                current_section = last_section['text']
        
        # Add question items with temporary section assignment (will be fixed by _assign_sections_by_question_below)
        for item in question_items:
            if item.get('tag') == '<Q>':
                item['form'] = current_form
                item['section'] = current_section  # Temporary - will be properly assigned later
            elif item.get('tag') == '<INPUT>':
                item['form'] = current_form
                item['section'] = current_section  # Temporary - will be properly assigned later
            
            all_items.append(item)
        
        # Match items with bounding boxes from OCR
        matched_items = self._match_items_to_lines(all_items, lines)
        
        # Assign proper question IDs based on position
        logger.info(f"[Page {page_num + 1}] Assigning question IDs based on position...")
        matched_items = self._assign_question_ids(matched_items)
        
        # Assign sections based on question_below_text from section headers
        logger.info(f"[Page {page_num + 1}] Assigning sections based on question_below_text...")
        matched_items = self._assign_sections_by_question_below(matched_items, current_section)
        
        # Update current_section to the last section found
        for item in reversed(matched_items):
            if item.get('tag') == '<SH>' and item.get('text'):
                current_section = item['text']
                break
        
        # Clear memory cache after page processing
        if self.clear_cache_per_page and self.device == "cuda":
            torch.cuda.empty_cache()
        
        return matched_items, current_form, current_section
    
    def _extract_questions(self, page_image: Image.Image, page_num: int) -> List[Dict]:
        """
        Step 1: Extract only question-input pairs from the page
        """
        prompt = """Extract ALL questions and their input fields from this CRF page.
        YOU MUST SCAN THE PAGE FROM TOP TO BOTTOM CAREFULLY. Do not miss any detail 
Hint: questions usually share same format and font size, if something looks different than other questions in format, font size or colour, it is not a question

OUTPUT FORMAT:
- For each question, output: <Q> followed by the question text
- For each input field, output: <INPUT> followed by the input text
- Input fields should immediately follow their question
- One item per line
- You must only output questions

EXAMPLE:
<Q> Date of visit
<INPUT> dd-mmm-yyyy
<Q> Is the subject eligible?
<INPUT> Yes
<INPUT> No
<Q> Subject ID
<INPUT> _______________

Extract everything - questions, checkboxes, text fields, date fields, etc.
Output ONLY the tagged lines, nothing else:"""

        try:
            response = self._query_model(page_image, prompt)
            logger.debug(f"Raw model output (questions extraction): {response}")
            items = self._parse_simple_tags(response, page_num, 'questions')
            return items
        except Exception as e:
            logger.error(f"Error extracting questions: {e}")
            return []
    
    def _extract_sections(self, page_image: Image.Image, page_num: int, 
                            question_items: List[Dict], form_items: List[Dict]) -> List[Dict]:
        """
        Step 3: Extract only section headers from the page (excluding form name and questions)
        """
        # Build list of already extracted question texts to exclude
        extracted_questions = []
        for item in question_items:
            if item.get('tag') == '<Q>' and item.get('text'):
                extracted_questions.append(item['text'])
        
        questions_list = "\n".join(f"- {q}" for q in extracted_questions[:10])  # Show first 10
        if len(extracted_questions) > 10:
            questions_list += f"\n... and {len(extracted_questions) - 10} more questions"
        
        # Extract form name to exclude
        form_name = None
        for item in form_items:
            if item.get('tag') == '<FORM>' and item.get('text'):
                form_name = item['text']
                break
        
        prompt = f"""Extract ONLY section headers from this CRF page and identify the first question below each section.

A section header is:
- A title for a group of related questions
- Usually in bold, larger font, or different color
- Examples: Demographics, Medical History, Vital Signs, Laboratory Tests

NOT a section header:
- Questions (anything ending with ?)
- Input fields or options
- The form name: {form_name if form_name else "(none found)"}
- These already found questions:
{questions_list if extracted_questions else "(none found)"}

YOU MUST NOT OUTPUT THE FOUND QUESTIONS AND FOUND FORM NAME
YOU MUST TO LOOK ALL OTHER CONTENT IN PAGE, FIND ALL POSSIBLE SECTION HEADER, BUT IT'S OK NOT FIND ONE
YOU MUST ONLY PARSE WHOLE TEXT, DO NOT PARSE PART OF IT
YOU MUST SCAN THE PAGE FROM TOP TO BOTTOM CAREFULLY. Do not miss any detail 

For each section header, identify the FIRST question that appears below it on the page.

OUTPUT FORMAT:
- For each section header, output: <SH> followed by the header text, then a pipe |, then the first question below it (or "none" if no question below)
- If no section headers found, output: SH_NOT_FOUND
- One item per line

EXAMPLE:
<SH> Demographics | Date of birth (DOB)
<SH> Inclusion Criteria | Age ≥ 18 years (INC1)
<SH> Physical Examination | none

Output ONLY the tagged lines or SH_NOT_FOUND:"""

        try:
            response = self._query_model(page_image, prompt)
            logger.debug(f"Raw model output (sections extraction): {response}")
            items = self._parse_simple_tags(response, page_num, 'sections')
            return items
        except Exception as e:
            logger.error(f"Error extracting sections: {e}")
            return []
    
    def _extract_form_name(self, page_image: Image.Image, page_num: int) -> List[Dict]:
        """
        Step 2: Extract only the form name from the page
        """
        prompt = """Find the main FORM NAME on this CRF page.

The form name is the overall title of the entire form.
Examples: Baseline Visit, Screening Visit, Follow-up Visit Week 12, Enrollment Form

Look for:
- Title at the top of the page
- Text in header/footer
- Largest title on the page

OUTPUT FORMAT:
- Output: <FORM> followed by the form name
- If no form name found, output: FORM_NOT_FOUND
- Only ONE form name per page

EXAMPLE:
<FORM> Baseline Visit

Output ONLY the tagged line or FORM_NOT_FOUND:"""

        try:
            response = self._query_model(page_image, prompt)
            logger.debug(f"Raw model output (form name extraction): {response}")
            items = self._parse_simple_tags(response, page_num, 'form')
            return items
        except Exception as e:
            logger.error(f"Error extracting form name: {e}")
            return []
    
    def _query_model(self, page_image: Image.Image, prompt: str) -> str:
        """
        Query the model with an image and prompt
        """
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a precise data extractor. Output only the requested format."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": page_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Generate with controlled temperature
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process inputs
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[text],
                images=[page_image],
                return_tensors="pt"
            )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return response
    
    def _parse_json_lines(self, response: str) -> List[Dict]:
        """
        Parse JSON Lines response into list of dicts with enhanced error handling
        """
        items = []
        
        # Check for special responses
        if "SH_NOT_FOUND" in response:
            logger.debug("No section headers found on this page")
            return []
        
        # Clean the response - remove any explanatory text before/after JSON
        lines = response.strip().split('\n')
        json_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line looks like JSON (starts with { and ends with })
            if line.startswith('{') and line.endswith('}'):
                json_lines.append(line)
            # Also try to extract JSON from lines that have JSON embedded
            elif '{' in line and '}' in line:
                try:
                    # Extract JSON part
                    start = line.index('{')
                    end = line.rindex('}') + 1
                    json_part = line[start:end]
                    json_lines.append(json_part)
                except ValueError:
                    continue
        
        # Parse each JSON line
        for line in json_lines:
            try:
                obj = json.loads(line)
                
                # Validate required fields based on tag
                tag = obj.get('tag')
                if tag == '<Q>':
                    if 'qid' not in obj:
                        # Use temporary ID for now - will be assigned properly later
                        obj['temp_qid'] = f"temp_q_{len([i for i in items if i.get('tag') == '<Q>'])}"
                        logger.debug(f"Question missing qid, assigned temp ID: {obj.get('text', '')[:50]}...")
                elif tag == '<INPUT>':
                    if 'parent_qid' not in obj:
                        # Use temporary parent ID for now
                        obj['parent_temp_qid'] = 'orphan'
                        logger.debug(f"Input missing parent_qid: {obj.get('text', '')[:50]}...")
                    if 'input_type' not in obj:
                        logger.warning(f"Input missing input_type: {obj.get('text', '')[:50]}...")
                        continue
                
                # Update sequence counter
                if 'seq' in obj:
                    self.current_seq = max(self.current_seq, obj['seq'] + 1)
                
                items.append(obj)
                
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON line: {line[:100]}... Error: {e}")
                continue
        
        # Log parsing results
        if items:
            tags_count = {}
            for item in items:
                tag = item.get('tag', 'unknown')
                tags_count[tag] = tags_count.get(tag, 0) + 1
            logger.debug(f"Parsed {len(items)} items: {tags_count}")
        else:
            logger.debug("No valid JSON items parsed from response")
        
        return items
    
    def _parse_simple_tags(self, response: str, page_num: int, tag_type: str) -> List[Dict]:
        """
        Parse simple tagged output from model
        
        Args:
            response: Model output with simple tags
            page_num: Current page number
            tag_type: Type of extraction ('questions', 'sections', 'form')
            
        Returns:
            List of parsed items as dicts
        """
        items = []
        lines = response.strip().split('\n')
        
        # Track question IDs for linking inputs - will be assigned later after sorting
        temp_qid_counter = 0
        temp_qid_map = {}  # Maps temporary IDs to items
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for special responses
            if tag_type == 'sections' and line == 'SH_NOT_FOUND':
                logger.debug("No section headers found on this page")
                return []
            elif tag_type == 'form' and line == 'FORM_NOT_FOUND':
                logger.debug("No form name found on this page")
                return []
                
            # Parse tagged line
            if line.startswith('<Q>'):
                text = line[3:].strip()
                # Use temporary ID that will be replaced later
                temp_qid = f"temp_q_{temp_qid_counter}"
                temp_qid_counter += 1
                
                item = {
                    'seq': self.current_seq,
                    'tag': '<Q>',
                    'text': text,
                    'page': page_num + 1,
                    'temp_qid': temp_qid  # Temporary ID
                }
                self.current_seq += 1
                items.append(item)
                temp_qid_map[temp_qid] = item
                
            elif line.startswith('<INPUT>'):
                text = line[7:].strip()
                
                # Determine input type based on content
                input_type = 'text'  # default
                text_lower = text.lower()
                
                if text_lower in ['yes', 'no', 'n/a', 'unknown', 'not done']:
                    input_type = 'option'
                elif any(marker in text_lower for marker in ['___', 'dd-mmm-yyyy', 'hh:mm', '/___/', '|__|']):
                    input_type = 'text'
                elif len(text) > 50:  # Long text likely free form
                    input_type = 'free'
                elif text.startswith('□') or text.startswith('○'):
                    input_type = 'option'
                    
                # Get the most recent question's temp ID
                recent_temp_qid = None
                for i in range(len(items) - 1, -1, -1):
                    if items[i].get('tag') == '<Q>':
                        recent_temp_qid = items[i].get('temp_qid')
                        break
                    
                item = {
                    'seq': self.current_seq,
                    'tag': '<INPUT>',
                    'text': text,
                    'page': page_num + 1,
                    'parent_temp_qid': recent_temp_qid or 'orphan',  # Temporary parent ID
                    'input_type': input_type
                }
                self.current_seq += 1
                items.append(item)
                
            elif line.startswith('<SH>'):
                # Parse section header with optional question below
                content = line[4:].strip()
                
                # Check if there's a pipe separator indicating question below
                if '|' in content:
                    parts = content.split('|', 1)
                    text = parts[0].strip()
                    question_below = parts[1].strip()
                else:
                    text = content
                    question_below = "none"
                
                item = {
                    'seq': self.current_seq,
                    'tag': '<SH>',
                    'text': text,
                    'page': page_num + 1,
                    'question_below_text': question_below
                }
                self.current_seq += 1
                items.append(item)
                
            elif line.startswith('<FORM>'):
                text = line[6:].strip()
                item = {
                    'seq': self.current_seq,
                    'tag': '<FORM>',
                    'text': text,
                    'page': page_num + 1
                }
                self.current_seq += 1
                items.append(item)
                
        # Log parsing results
        if items:
            tags_count = {}
            for item in items:
                tag = item.get('tag', 'unknown')
                tags_count[tag] = tags_count.get(tag, 0) + 1
            logger.debug(f"Parsed {len(items)} items: {tags_count}")
        
        return items
    
    def _assign_question_ids(self, items: List[Dict]) -> List[Dict]:
        """
        Assign question IDs to questions based on their position on the page.
        Questions are numbered sequentially in top-to-bottom, left-to-right order.
        Also improves option-to-question assignment based on proximity.
        
        Args:
            items: List of items with bbox information
            
        Returns:
            Updated items with proper question IDs and improved parent assignments
        """
        # Separate items by type
        questions = []
        inputs = []
        other_items = []
        
        for item in items:
            if item.get('tag') == '<Q>' and 'bbox_xyxy' in item:
                questions.append(item)
            elif item.get('tag') == '<INPUT>' and 'bbox_xyxy' in item:
                inputs.append(item)
            else:
                other_items.append(item)
        
        # Sort questions by sequence number instead of position
        questions.sort(key=lambda x: x.get('seq', 0))
        
        # Assign question IDs using global counter based on seq order
        temp_to_final_qid = {}
        for i, question in enumerate(questions):
            final_qid = f"q-{self.global_question_counter:03d}"  # q-000, q-001, q-002, etc.
            temp_qid = question.get('temp_qid')
            if temp_qid:
                temp_to_final_qid[temp_qid] = final_qid
                del question['temp_qid']  # Remove temporary ID
            question['qid'] = final_qid
            logger.debug(f"Assigned {final_qid} to question (seq={question.get('seq', 0)}): {question['text'][:50]}...")
            self.global_question_counter += 1
        
        # Sort inputs by position for better assignment
        inputs.sort(key=lambda x: (x['bbox_xyxy'][1], x['bbox_xyxy'][0]))
        
        # Update parent_qid for inputs with improved logic
        for input_item in inputs:
            parent_temp_qid = input_item.get('parent_temp_qid')
            
            # First try to use the temporary mapping
            if parent_temp_qid and parent_temp_qid in temp_to_final_qid:
                input_item['parent_qid'] = temp_to_final_qid[parent_temp_qid]
                if 'parent_temp_qid' in input_item:
                    del input_item['parent_temp_qid']
            else:
                # Find the closest question above and to the left of this input
                input_y = input_item['bbox_xyxy'][1]
                input_x = input_item['bbox_xyxy'][0]
                
                best_qid = None
                best_score = float('inf')
                
                for question in questions:
                    q_y = question['bbox_xyxy'][1]
                    q_x = question['bbox_xyxy'][0]
                    
                    # Question must be above or at same level as input
                    if q_y <= input_y:
                        # Calculate distance with preference for questions directly above
                        y_distance = input_y - q_y
                        x_distance = abs(input_x - q_x)
                        
                        # Score based on Y distance primarily, with X distance as tiebreaker
                        # Add penalty for questions too far to the left or right
                        score = y_distance + (x_distance * 0.1 if x_distance < 200 else x_distance)
                        
                        # Special handling for options that are typically indented
                        if input_item.get('input_type') == 'option' and x_distance < 100:
                            score *= 0.8  # Prefer nearby questions for options
                        
                        if score < best_score:
                            best_score = score
                            best_qid = question['qid']
                
                # Assign the best match or mark as orphan
                if best_qid and best_score < 500:  # Reasonable distance threshold
                    input_item['parent_qid'] = best_qid
                    logger.debug(f"Assigned input '{input_item['text'][:30]}...' to question {best_qid} (score: {best_score:.1f})")
                else:
                    input_item['parent_qid'] = 'orphan'
                    logger.debug(f"Input '{input_item['text'][:30]}...' marked as orphan (best score: {best_score:.1f})")
                
                if 'parent_temp_qid' in input_item:
                    del input_item['parent_temp_qid']
        
        # Combine all items back together
        all_items = questions + inputs + other_items
        
        # Sort all items by sequence number for final output
        all_items.sort(key=lambda x: x.get('seq', 0))
        
        # Log assignment summary
        orphan_count = sum(1 for item in inputs if item.get('parent_qid') == 'orphan')
        if orphan_count > 0:
            logger.warning(f"Found {orphan_count} orphan inputs without parent questions")
        
        return all_items
    
    def _clean_similar_sections(self, section_items: List[Dict], question_items: List[Dict], 
                               similarity_threshold: float = 0.9) -> List[Dict]:
        """
        Remove section headers that are too similar to questions or input options
        
        Args:
            section_items: List of section header items
            question_items: List of question and input items
            similarity_threshold: Threshold for considering items as too similar (0.9 = 90%)
            
        Returns:
            Cleaned list of section items
        """
        cleaned_sections = []
        
        # Extract question texts for comparison
        question_texts = [q.get('text', '').strip().lower() 
                         for q in question_items 
                         if q.get('tag') == '<Q>' and q.get('text')]
        
        # Also extract input/option texts for comparison
        input_texts = [i.get('text', '').strip().lower() 
                      for i in question_items 
                      if i.get('tag') == '<INPUT>' and i.get('text')]
        
        # Combine all texts to check against
        all_texts_to_check = question_texts + input_texts
        
        for section in section_items:
            if section.get('tag') != '<SH>' or not section.get('text'):
                cleaned_sections.append(section)
                continue
                
            section_text = section['text'].strip().lower()
            is_too_similar = False
            
            # Check similarity with each question or input text
            for text in all_texts_to_check:
                # Use SequenceMatcher for similarity calculation
                similarity = SequenceMatcher(None, section_text, text).ratio()
                
                # Only check similarity ratio, not substring
                if similarity >= similarity_threshold:
                    text_type = "question" if text in question_texts else "input/option"
                    logger.debug(f"Removing section '{section['text']}' - too similar to {text_type} '{text}' (similarity: {similarity:.2f})")
                    is_too_similar = True
                    break
            
            if not is_too_similar:
                cleaned_sections.append(section)
            else:
                logger.info(f"Filtered out section header '{section['text']}' as it's too similar to a question")
                
        return cleaned_sections
    
    def _assign_sections_by_question_below(self, items: List[Dict], current_section: Optional[str] = None) -> List[Dict]:
        """
        Assign sections to questions based on the question_below_text field in section headers.
        
        Args:
            items: List of items sorted by seq
            current_section: Section from previous page
            
        Returns:
            Updated items with section assignments
        """
        # First, create a mapping of question text to qid
        question_text_to_qid = {}
        for item in items:
            if item.get('tag') == '<Q>' and 'qid' in item:
                question_text_to_qid[item['text']] = item['qid']
        
        # Find all section headers
        sections = []
        for item in items:
            if item.get('tag') == '<SH>':
                sections.append(item)
        
        # Sort sections by seq
        sections.sort(key=lambda x: x.get('seq', 0))
        
        # Create section ranges based on question_below_text
        section_ranges = []
        for i, section in enumerate(sections):
            section_name = section['text']
            section_seq = section['seq']
            question_below_text = section.get('question_below_text', 'none')
            
            # The section starts at the first question below it (or right after the header if no question specified)
            if question_below_text != 'none':
                # Try to find the question by text
                start_seq = None
                for item in items:
                    if item.get('tag') == '<Q>' and item.get('seq', 0) > section_seq:
                        # Must be after the section header
                        if item['text'] == question_below_text or question_below_text in item['text']:
                            start_seq = item['seq']
                            break
                
                # If we couldn't find the specific question, start right after the section header
                if start_seq is None:
                    start_seq = section_seq + 0.1  # Small offset to include items right after the header
            else:
                # No specific question mentioned, section starts right after the header
                start_seq = section_seq + 0.1
            
            # Find the end of this section (start of next section)
            if i + 1 < len(sections):
                next_section = sections[i + 1]
                # The current section ends where the next section's range would start
                # For now, we'll use the next section header's position
                end_seq = next_section['seq']
            else:
                # Last section goes to the end
                end_seq = float('inf')
            
            section_ranges.append({
                'name': section_name,
                'section_seq': section_seq,
                'start_seq': start_seq,
                'end_seq': end_seq
            })
        
        # Assign sections to questions based on their position
        for item in items:
            if item.get('tag') == '<Q>':
                item_seq = item.get('seq', 0)
                assigned_section = current_section  # Default to inherited section
                
                # For questions before all sections, use inherited section
                if sections and item_seq < sections[0]['seq']:
                    assigned_section = current_section
                else:
                    # Check each section range
                    for section_range in section_ranges:
                        # A question belongs to a section if:
                        # 1. It appears after the section header (seq > section_seq)
                        # 2. It appears before the next section header (seq < end_seq)
                        if section_range['section_seq'] < item_seq < section_range['end_seq']:
                            # Additional check: if this section has a specific start_seq,
                            # only assign if the question is at or after that seq
                            if item_seq >= section_range['start_seq']:
                                assigned_section = section_range['name']
                                break
                
                # Always update the section, overriding any previous assignment
                item['section'] = assigned_section
                
            elif item.get('tag') == '<INPUT>':
                # Inputs inherit section from their parent question
                parent_qid = item.get('parent_qid')
                if parent_qid:
                    # Find parent question's section
                    for q_item in items:
                        if q_item.get('tag') == '<Q>' and q_item.get('qid') == parent_qid:
                            item['section'] = q_item.get('section', current_section)
                            break
                else:
                    item['section'] = current_section
        
        return items
    
    def _add_question_positions_to_sections(self, items: List[Dict]) -> List[Dict]:
        """
        Add question_above and question_below fields to section headers.
        These fields contain the qid of the nearest questions.
        
        Args:
            items: List of items sorted by seq
            
        Returns:
            Updated items with question position info for sections
        """
        # First pass: identify all questions and sections with their seq numbers
        questions = []
        sections = []
        
        for item in items:
            if item.get('tag') == '<Q>' and 'qid' in item:
                questions.append({
                    'seq': item.get('seq', 0),
                    'qid': item['qid'],
                    'index': items.index(item)
                })
            elif item.get('tag') == '<SH>':
                sections.append({
                    'seq': item.get('seq', 0),
                    'index': items.index(item),
                    'item': item
                })
        
        # Sort by seq to ensure proper ordering
        questions.sort(key=lambda x: x['seq'])
        sections.sort(key=lambda x: x['seq'])
        
        # For each section, find nearest questions
        for section_info in sections:
            section_seq = section_info['seq']
            section_item = section_info['item']
            
            # Find question above (largest seq that is still smaller than section seq)
            question_above = None
            for q in reversed(questions):
                if q['seq'] < section_seq:
                    question_above = q['qid']
                    break
            
            # Find question below (smallest seq that is larger than section seq)
            question_below = None
            for q in questions:
                if q['seq'] > section_seq:
                    question_below = q['qid']
                    break
            
            # Update section item
            section_item['question_above'] = question_above if question_above else "none"
            section_item['question_below'] = question_below if question_below else "none"
            
            logger.debug(f"Section '{section_item['text']}' (seq={section_seq}): "
                        f"question_above={section_item['question_above']}, "
                        f"question_below={section_item['question_below']}")
        
        return items
    
    def _reassign_sections_by_bbox(self, all_items: List[Dict], current_section: Optional[str] = None) -> List[Dict]:
        """
        Reassign sections to questions based on their bounding box positions.
        Questions below a section header belong to that section.
        
        Args:
            all_items: List of all items with bbox information
            current_section: Section name from previous page
            
        Returns:
            Updated items with corrected section assignments
        """
        # Extract current form from items
        current_form = None
        for item in all_items:
            if item.get('tag') == '<FORM>' and item.get('text'):
                current_form = item['text']
                break
        
        # Separate items by type
        section_headers = []
        questions = []
        inputs = []
        forms = []
        
        for item in all_items:
            if item.get('tag') == '<SH>' and 'bbox_xyxy' in item:
                section_headers.append(item)
            elif item.get('tag') == '<Q>' and 'bbox_xyxy' in item:
                questions.append(item)
            elif item.get('tag') == '<INPUT>' and 'bbox_xyxy' in item:
                inputs.append(item)
            elif item.get('tag') == '<FORM>':
                forms.append(item)
        
        # Sort all items by Y position (top to bottom) then X position (left to right)
        section_headers.sort(key=lambda x: (x['bbox_xyxy'][1], x['bbox_xyxy'][0]))
        questions.sort(key=lambda x: (x['bbox_xyxy'][1], x['bbox_xyxy'][0]))
        inputs.sort(key=lambda x: (x['bbox_xyxy'][1], x['bbox_xyxy'][0]))
        
        # Create section ranges based on Y positions
        section_ranges = []
        for i, section in enumerate(section_headers):
            start_y = section['bbox_xyxy'][1]
            # End Y is either the next section's Y or infinity
            end_y = section_headers[i + 1]['bbox_xyxy'][1] if i + 1 < len(section_headers) else float('inf')
            section_ranges.append({
                'section': section['text'],
                'start_y': start_y,
                'end_y': end_y
            })
        
        # Assign sections to questions based on their Y position
        section_assignments = {}
        
        for question in questions:
            q_y = question['bbox_xyxy'][1]  # Y position of question
            assigned_section = current_section  # Default to inherited section
            
            # Find which section range this question falls into
            for section_range in section_ranges:
                if section_range['start_y'] <= q_y < section_range['end_y']:
                    assigned_section = section_range['section']
                    break
            
            # For questions above all sections, use inherited section
            if section_headers and q_y < section_headers[0]['bbox_xyxy'][1]:
                assigned_section = current_section
            
            # Store assignment
            qid = question.get('qid') or question.get('temp_qid')
            if qid:
                section_assignments[qid] = assigned_section
                logger.debug(f"Question {qid} at Y={q_y:.3f} assigned to section: {assigned_section}")
        
        # Build final items list with proper sections
        updated_items = []
        
        # Add forms first
        updated_items.extend(forms)
        
        # Add section headers
        updated_items.extend(section_headers)
        
        # Add questions with assigned sections
        for question in questions:
            qid = question.get('qid') or question.get('temp_qid')
            if qid and qid in section_assignments:
                question['section'] = section_assignments[qid]
            else:
                question['section'] = current_section
            # Ensure form is set
            if 'form' not in question or not question['form']:
                question['form'] = current_form
            updated_items.append(question)
        
        # Add inputs with their parent question's section
        for input_item in inputs:
            parent_qid = input_item.get('parent_qid') or input_item.get('parent_temp_qid')
            if parent_qid and parent_qid in section_assignments:
                input_item['section'] = section_assignments[parent_qid]
            else:
                # Try to find section based on position
                input_y = input_item['bbox_xyxy'][1]
                assigned_section = current_section
                for section_range in section_ranges:
                    if section_range['start_y'] <= input_y < section_range['end_y']:
                        assigned_section = section_range['section']
                        break
                input_item['section'] = assigned_section
            # Ensure form is set
            if 'form' not in input_item or not input_item['form']:
                input_item['form'] = current_form
            updated_items.append(input_item)
        
        # Sort all items by sequence number for final output
        updated_items.sort(key=lambda x: x.get('seq', 0))
        
        # Log section assignment summary
        if section_assignments:
            sections_used = set(section_assignments.values())
            logger.info(f"Assigned {len(section_assignments)} questions to {len(sections_used)} sections")
        
        return updated_items
    
    def _match_items_to_lines(self, items: List[Dict], lines: List[ExtractedLine]) -> List[Dict]:
        """
        Match extracted items to OCR lines for bounding boxes
        """
        def _best_match_line(text: str) -> Tuple[Optional[ExtractedLine], int]:
            """Find OCR line that best matches the text."""
            base = text.lower().strip()
            best, score = None, 0
            base_tokens = set(base.split())
            for ln in lines:
                ln_low = ln.text.lower()
                if base in ln_low or ln_low in base:
                    s = len(base_tokens & set(ln_low.split()))
                    if s > score:
                        best, score = ln, s
            return best, score
        
        matched_items = []
        for item in items:
            text = item.get('text', '')
            best_line, _ = _best_match_line(text)
            
            if best_line:
                item['bbox_xyxy'] = best_line.bbox.to_list()
                item['bbox_ref'] = best_line.line_id
            else:
                item['bbox_xyxy'] = [0, 0, 1, 1]
                item['bbox_ref'] = "unmatched"
            
            matched_items.append(item)
        
        return matched_items
            
    def _parse_structured_response(
        self,
        response: str,
        lines: List[ExtractedLine],
        page_num: int,
        current_form: Optional[str] = None,
        initial_section: Optional[str] = None,
        allow_empty: bool = False,
    ) -> Tuple[List[Dict], Optional[str], Optional[str]]:
        import re

        import json
        
        def _best_match_line(text: str) -> Tuple[Optional[ExtractedLine], int]:
            """Find OCR line that best matches the text."""
            base = text.lower().strip()
            best, score = None, 0
            base_tokens = set(base.split())
            for ln in lines:
                ln_low = ln.text.lower()
                if base in ln_low or ln_low in base:
                    s = len(base_tokens & set(ln_low.split()))
                    if s > score:
                        best, score = ln, s
            return best, score
        
        # Parse JSON Lines
        items = []
        form_name = current_form
        section = initial_section
        
        # Split by lines and parse each JSON object
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse JSON line: {line[:100]}...")
                continue
            
            # Fix common model mistakes
            if obj.get('tag') == '<Q>' and 'parent_qid' in obj and 'qid' not in obj:
                # Model put parent_qid instead of qid on a question
                obj['qid'] = obj.pop('parent_qid')
                logger.debug(f"Fixed question with parent_qid -> qid: {obj.get('text', '')[:50]}...")
            
            # Update seq counter
            if 'seq' in obj:
                self.current_seq = max(self.current_seq, obj['seq'] + 1)
            
            # Find matching line for bbox
            text = obj.get('text', '')
            best_line, _ = _best_match_line(text)
            
            # Add bbox info to all items
            if best_line:
                obj['bbox_xyxy'] = best_line.bbox.to_list()
                obj['bbox_ref'] = best_line.line_id
            else:
                obj['bbox_xyxy'] = [0, 0, 1, 1]
                obj['bbox_ref'] = "unmatched"
            
            # Track form and section
            if obj.get('tag') == '<FORM>':
                form_name = text
                obj['is_form'] = True
            elif obj.get('tag') == '<SH>':
                section = text
                obj['is_section'] = True
            
            # Add form/section context to questions and inputs
            if obj.get('tag') == '<Q>':
                obj['form'] = form_name
                obj['section'] = section
            elif obj.get('tag') == '<INPUT>':
                obj['form'] = form_name
                obj['section'] = section
            
            items.append(obj)
        
        if items:
            logger.info(f"Extracted {len(items)} items from JSON Lines")
            forms = [i['text'] for i in items if i.get('tag') == '<FORM>']
            sections = [i['text'] for i in items if i.get('tag') == '<SH>']
            questions = [i for i in items if i.get('tag') == '<Q>']
            inputs = [i for i in items if i.get('tag') == '<INPUT>']
            
            if forms:
                logger.info(f"Forms: {', '.join(forms)}")
            if sections:
                logger.info(f"Sections: {', '.join(sections)}")
            logger.info(f"Questions: {len(questions)}, Inputs: {len(inputs)}")
        elif not allow_empty:
            logger.debug(f"No items found in response: {response[:200]}...")
            
        return items, form_name, section
            
    def _fallback_extraction(self, lines: List[ExtractedLine], page_num: int, current_section: Optional[str] = None) -> List[CRFItem]:
        """Conservative fallback extraction - only extract clear questions"""
        logger.info("Using fallback extraction method")
        items = []
        
        # Parse current section into main and subsection
        section = None
        subsection = None
        if current_section:
            parts = current_section.split(" > ")
            section = parts[0]
            subsection = parts[1] if len(parts) > 1 else None
        
        for i, line in enumerate(lines):
            text = line.text.strip()
            if not text:
                continue
                
            # Check for section headers
            section_patterns = [
                "Inclusion Criteria", "Exclusion Criteria",
                "Demographics", "Medical History", "Vital Signs",
                "Physical Exam", "Laboratory", "Adverse Events",
                "Concomitant Medications", "Eligibility"
            ]
            
            for pattern in section_patterns:
                if pattern.lower() in text.lower():
                    section = pattern
                    logger.info(f"Found section: {section}")
                    break
                
            # Only extract items that are clearly questions
            is_question = False
            item_type = "question"
            
            # Strong indicators of questions
            if text.endswith("?"):
                is_question = True
            elif text.endswith(":") and len(text) > 5:
                # Check if it's followed by checkboxes or radio buttons
                if i + 1 < len(lines):
                    next_text = lines[i + 1].text
                    if any(cb in next_text for cb in ["□", "☐", "[ ]", "○", "◯", "( )"]):
                        is_question = True
                        item_type = "checkbox" if "□" in next_text or "☐" in next_text else "radio"
            elif any(kw in text.lower() for kw in ["please select", "please enter", "please specify", "check all"]):
                is_question = True
                
            if is_question:
                item = CRFItem(
                    item_id=f"p{page_num + 1}_i{i + 1}",
                    type=item_type,
                    text=text,
                    bbox_xyxy=line.bbox.to_list(),
                    bbox_ref=line.line_id,
                    options=None,
                    section=section
                )
                items.append(item)
            
        return items
    
    def _digitize_with_custom_prompt(self, page_image: Image.Image, lines: List[ExtractedLine], 
                                     page_num: int, custom_prompt: str) -> List[CRFItem]:
        """
        Digitize a page using a custom prompt
        
        Args:
            page_image: PIL Image of the page
            lines: Extracted text lines
            page_num: Page number
            custom_prompt: Custom prompt to use
            
        Returns:
            List of extracted CRF items
        """
        if self.fallback_only:
            return self._fallback_extraction(lines, page_num, None)
            
        # Prepare messages for Qwen-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": page_image},
                    {"type": "text", "text": custom_prompt}
                ]
            }
        ]
        
        # Generate with controlled temperature
        try:
            if hasattr(self, 'apply_chat_template') and callable(self.apply_chat_template):
                # New processor API
                text = self.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            elif hasattr(self.processor, 'apply_chat_template'):
                # Standard API
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback for older versions
                text = custom_prompt
                
            # Get model inputs
            if hasattr(self, '__call__'):
                inputs = self(text=text, images=[page_image], return_tensors="pt", padding=True).to(self.device)
            else:
                inputs = self.processor(text=text, images=[page_image], return_tensors="pt", padding=True).to(self.device)
            
            # Generate with deterministic parameters
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False
            )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the assistant's response
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            elif text in response:
                response = response.replace(text, "").strip()
                
            logger.debug(f"[Custom prompt] Raw model output:")
            logger.debug("-" * 80)
            logger.debug(response)
            logger.debug("-" * 80)
            
            # Check for no questions response
            if "NO_QUESTIONS_FOUND" in response:
                logger.info("Model indicates no question-option pairs on this page")
                return []
                
            # Parse structured response (without section tracking for re-examination)
            items, _, _ = self._parse_structured_response(response, lines, page_num, None, None, allow_empty=True)
            
            return items
            
        except Exception as e:
            logger.error(f"Error in custom prompt digitization: {str(e)}")
            logger.info("Falling back to standard extraction")
            return self._fallback_extraction(lines, page_num, None)
    
    def _format_line_context(self, lines: List[ExtractedLine]) -> str:
        """Format extracted lines for prompt context"""
        context_lines = []
        for line in lines:
            bbox_str = f"[{line.bbox.x1:.3f}, {line.bbox.y1:.3f}, {line.bbox.x2:.3f}, {line.bbox.y2:.3f}]"
            context_lines.append(f"{line.line_id}: {line.text} | bbox: {bbox_str}")
        return "\n".join(context_lines[:50])  # Limit to prevent token overflow
        
    def process_pdf(self, pdf_path: Path, doc_id: Optional[str] = None, 
                   enable_qc: bool = True, max_pages: Optional[int] = None) -> Dict:
        """
        Process entire PDF document
        
        Args:
            pdf_path: Path to CRF PDF
            doc_id: Document identifier (defaults to filename)
            enable_qc: Enable quality control and re-examination of outlier pages
            max_pages: Maximum number of pages to process (None for all)
            
        Returns:
            Processing summary with output paths
        """
        pdf_path = Path(pdf_path)
        if not doc_id:
            doc_id = pdf_path.stem
            
        logger.info(f"Processing CRF: {pdf_path.name}")
        
        # Reset global counters for new document
        self.global_question_counter = 0
        
        # Start total timer
        total_start_time = time.time()
        
        # Create output directory for this document
        doc_output_dir = self.output_dir / doc_id
        doc_output_dir.mkdir(exist_ok=True)
        
        # Open PDF
        doc = fitz.open(pdf_path)
        num_pages = doc.page_count
        doc.close()
        
        # Process each page
        all_items = []
        page_results = []
        page_times = []
        question_counts = []  # Track questions per page for quality control
        current_form = None  # Track form name across pages
        current_section = None  # Track section across pages
        
        # First pass - process all pages
        pages_to_process = min(num_pages, max_pages) if max_pages else num_pages
        for page_num in range(pages_to_process):
            page_start_time = time.time()
            logger.info(f"Processing page {page_num + 1}/{pages_to_process} (of {num_pages} total)")
            
            # Extract primitives
            page_image, lines = self.extract_page_primitives(pdf_path, page_num)
            
            # Save line data for reference
            lines_data = [
                {
                    "line_id": line.line_id,
                    "text": line.text,
                    "bbox": line.bbox.to_list(),
                    "page_num": line.page_num
                }
                for line in lines
            ]
            
            # Digitize with Qwen-VL
            items, current_form, current_section = self.digitize_page(page_image, lines, page_num, current_form, current_section)
            
            # Extract supplemental page data (non-question text)
            supplemental_data = []
            
            if items:
                # If we have questions, supplemental data is everything NOT in questions/inputs
                question_texts = {item.get('text', '').lower() for item in items if item.get('tag') in ['<Q>', '<INPUT>']}
                
                for line in lines:
                    line_text_lower = line.text.lower().strip()
                    # Check if this line is NOT part of any extracted question
                    is_question = any(q_text in line_text_lower or line_text_lower in q_text 
                                    for q_text in question_texts)
                    
                    if not is_question and line.text.strip():
                        # Try to identify if this is a section header
                        text = line.text.strip()
                        supp_type = "supplemental"
                        
                        # Common section patterns
                        section_patterns = [
                            "Inclusion Criteria", "Exclusion Criteria",
                            "Demographics", "Medical History", "Vital Signs",
                            "Physical Exam", "Laboratory", "Adverse Events",
                            "Concomitant Medications", "Eligibility"
                        ]
                        
                        for pattern in section_patterns:
                            if pattern.lower() in text.lower():
                                supp_type = "section_header"
                                break
                        
                        supplemental_data.append({
                            "line_id": line.line_id,
                            "text": text,
                            "bbox": line.bbox.to_list(),
                            "type": supp_type
                        })
            else:
                # If no questions found, ALL text is supplemental
                for line in lines:
                    if line.text.strip():
                        supplemental_data.append({
                            "line_id": line.line_id,
                            "text": line.text.strip(),
                            "bbox": line.bbox.to_list(),
                            "type": "supplemental"
                        })
            
            # Create page result
            page_result = {
                "page_id": page_num,
                "items": items,  # Already dict format
                "supplemental_page_data": supplemental_data,
                "all_lines": lines_data,
                "metadata": {
                    "num_questions": len([i for i in items if i.get('tag') == '<Q>']),
                    "num_inputs": len([i for i in items if i.get('tag') == '<INPUT>']),
                    "num_inputs_by_type": {
                        "option": len([i for i in items if i.get('tag') == '<INPUT>' and i.get('input_type') == 'option']),
                        "text": len([i for i in items if i.get('tag') == '<INPUT>' and i.get('input_type') == 'text']),
                        "free": len([i for i in items if i.get('tag') == '<INPUT>' and i.get('input_type') == 'free'])
                    },
                    "num_forms": len([i for i in items if i.get('tag') == '<FORM>']),
                    "num_sections": len([i for i in items if i.get('tag') == '<SH>']),
                    "num_supplemental": len(supplemental_data),
                    "num_total_lines": len(lines),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
            
            # Save page JSON
            page_json_path = doc_output_dir / f"page_{page_num:03d}.json"
            with open(page_json_path, 'w') as f:
                json.dump(page_result, f, indent=2)
                
            page_results.append(page_json_path)
            all_items.extend(items)
            question_counts.append(len([i for i in items if i.get('tag') == '<Q>']))
            
            # Log page timing
            page_time = time.time() - page_start_time
            page_times.append(page_time)
            num_questions = len([i for i in items if i.get('tag') == '<Q>'])
            logger.info(f"Page {page_num + 1} completed in {page_time:.2f} seconds with {num_questions} questions")
            
        # Quality control - check for outlier pages
        pages_marked_unusual = []  # Track pages that need second inspection
        anomaly_reasons = {}  # Track why each page was marked unusual
        
        if enable_qc and len(question_counts) > 1:
            avg_questions = sum(question_counts) / len(question_counts)
            std_dev = (sum((x - avg_questions) ** 2 for x in question_counts) / len(question_counts)) ** 0.5
            logger.info(f"Average questions per page: {avg_questions:.1f} (std dev: {std_dev:.1f})")
            
            # Find pages that need re-examination
            pages_to_reexamine = []
            for page_num, count in enumerate(question_counts):
                reasons = []
                
                # Check for statistical outliers (more than 2 std devs away)
                if std_dev > 0 and abs(count - avg_questions) > 2 * std_dev:
                    reasons.append(f"statistical_outlier (count={count}, avg={avg_questions:.1f}, std={std_dev:.1f})")
                    pages_to_reexamine.append((page_num, count))
                
                # Check for extreme cases
                if count == 0:
                    reasons.append("zero_questions_found")
                    pages_to_reexamine.append((page_num, count))
                elif count > avg_questions * 2:
                    reasons.append("excessive_questions")
                    pages_to_reexamine.append((page_num, count))
                elif count < avg_questions * 0.3 and avg_questions > 5:
                    reasons.append("too_few_questions")
                    pages_to_reexamine.append((page_num, count))
                
                # Check page-specific anomalies
                page_json_path = doc_output_dir / f"page_{page_num:03d}.json"
                if page_json_path.exists():
                    with open(page_json_path, 'r') as f:
                        page_data = json.load(f)
                    
                    # Check for unusual supplemental data ratio
                    num_supplemental = page_data['metadata']['num_supplemental']
                    total_lines = page_data['metadata']['num_total_lines']
                    if total_lines > 0:
                        supp_ratio = num_supplemental / total_lines
                        if supp_ratio > 0.9 and count > 0:
                            reasons.append(f"high_supplemental_ratio ({supp_ratio:.2f})")
                        elif supp_ratio < 0.1 and count < avg_questions * 0.5:
                            reasons.append(f"low_supplemental_ratio ({supp_ratio:.2f})")
                    
                    # Check for parsing anomalies in questions
                    questions = page_data.get('question_option_pairs', [])
                    malformed_count = 0
                    for q in questions:
                        # Check for overly long sections (likely parsing error)
                        if q.get('section') and len(q['section']) > 50:
                            malformed_count += 1
                        # Check for missing critical fields
                        if not q.get('text') or not q.get('bbox_xyxy'):
                            malformed_count += 1
                    
                    if malformed_count > len(questions) * 0.3:
                        reasons.append(f"high_malformed_questions ({malformed_count}/{len(questions)})")
                
                if reasons:
                    pages_marked_unusual.append(page_num)
                    anomaly_reasons[page_num] = reasons
                    logger.warning(f"Page {page_num + 1} marked as unusual: {', '.join(reasons)}")
            
            # Re-examine outlier pages with more careful prompting
            if False and pages_to_reexamine and not self.fallback_only:  # TODO: Update re-examination for new format
                logger.info(f"Re-examining {len(pages_to_reexamine)} outlier pages...")
                
                for page_num, original_count in pages_to_reexamine:
                    logger.info(f"Re-examining page {page_num + 1} (originally found {original_count} questions)")
                    
                    # Re-extract primitives
                    page_image, lines = self.extract_page_primitives(pdf_path, page_num)
                    
                    # Use special prompt for re-examination
                    reexam_prompt = f"""QUALITY CHECK: Found {original_count} fields but expected ~{int(avg_questions)}.

Re-examine for missed fields in:
- Tables and grids
- Multi-column layouts
- Small text areas
- Time fields (need TWO lines)
- Numeric fields with units

REQUIRED FORMAT:
[SECTION: name] "question text (CODE)" (choices or placeholder)

Rules:
- NO SUBSECTIONS
- One field per line
- Section on EVERY line
- Include all field codes

Current section: {current_section if current_section else "NONE"}

Re-extract ALL fields:"""
                    
                    # Re-digitize with special attention
                    items = self._digitize_with_custom_prompt(page_image, lines, page_num, reexam_prompt)
                    
                    # Ensure items is a list of CRFItem objects
                    if not isinstance(items, list):
                        logger.error(f"Re-examination returned non-list: {type(items)}")
                        continue
                    
                    if len(items) != original_count:
                        logger.info(f"Re-examination found {len(items)} questions (was {original_count})")
                        
                        # Update the page result
                        page_json_path = doc_output_dir / f"page_{page_num:03d}.json"
                        
                        # Re-create the page result with new items
                        # (Reuse the supplemental data extraction logic)
                        supplemental_data = []
                        if items:
                            # Ensure items are CRFItem objects with text attribute
                            question_texts = set()
                            for item in items:
                                if isinstance(item, dict) and 'text' in item:
                                    question_texts.add(item['text'].lower())
                                elif hasattr(item, 'text'):
                                    question_texts.add(item.text.lower())
                                else:
                                    logger.warning(f"Item missing text attribute: {type(item)}")
                            for line in lines:
                                line_text_lower = line.text.lower().strip()
                                is_question = any(q_text in line_text_lower or line_text_lower in q_text 
                                                for q_text in question_texts)
                                
                                if not is_question and line.text.strip():
                                    text = line.text.strip()
                                    supp_type = "supplemental"
                                    
                                    section_patterns = [
                                        "Inclusion Criteria", "Exclusion Criteria",
                                        "Demographics", "Medical History", "Vital Signs",
                                        "Physical Exam", "Laboratory", "Adverse Events",
                                        "Concomitant Medications", "Eligibility"
                                    ]
                                    
                                    for pattern in section_patterns:
                                        if pattern.lower() in text.lower():
                                            supp_type = "section_header"
                                            break
                                    
                                    supplemental_data.append({
                                        "line_id": line.line_id,
                                        "text": text,
                                        "bbox": line.bbox.to_list(),
                                        "type": supp_type
                                    })
                        
                        lines_data = [
                            {
                                "line_id": line.line_id,
                                "text": line.text,
                                "bbox": line.bbox.to_list(),
                                "page_num": line.page_num
                            }
                            for line in lines
                        ]
                        
                        page_result = {
                            "page_id": page_num,
                            "question_option_pairs": [item.to_dict() for item in items],
                            "supplemental_page_data": supplemental_data,
                            "all_lines": lines_data,
                            "metadata": {
                                "num_questions": len(items),
                                "num_supplemental": len(supplemental_data),
                                "num_total_lines": len(lines),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "reexamined": True,
                                "original_count": original_count
                            }
                        }
                        
                        # Save updated page JSON
                        with open(page_json_path, 'w') as f:
                            json.dump(page_result, f, indent=2)
                        
                        # Update totals
                        # Remove old items from this page and add new ones
                        all_items = [item for item in all_items if not item.item_id.startswith(f"p{page_num + 1}_")]
                        all_items.extend(items)
                        question_counts[page_num] = len(items)
        
        # Create second inspection queue for unusual pages
        second_inspection_queue = []
        if enable_qc:
            for page_num in pages_marked_unusual:
                page_json_path = doc_output_dir / f"page_{page_num:03d}.json"
                if page_json_path.exists():
                    with open(page_json_path, 'r') as f:
                        page_data = json.load(f)
                    
                    # Mark page for second inspection
                    page_data['metadata']['requires_second_inspection'] = True
                    page_data['metadata']['anomaly_reasons'] = anomaly_reasons.get(page_num, [])
                    
                    # Save updated page data
                    with open(page_json_path, 'w') as f:
                        json.dump(page_data, f, indent=2)
                    
                    second_inspection_queue.append({
                        'page_num': page_num,
                        'page_file': str(page_json_path),
                        'anomaly_reasons': anomaly_reasons.get(page_num, []),
                        'question_count': question_counts[page_num]
                    })
        
        # Save second inspection queue
        if second_inspection_queue:
            queue_path = doc_output_dir / "second_inspection_queue.json"
            with open(queue_path, 'w') as f:
                json.dump({
                    'doc_id': doc_id,
                    'total_pages_marked': len(second_inspection_queue),
                    'pages': second_inspection_queue,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
            logger.info(f"Created second inspection queue with {len(second_inspection_queue)} pages")
        
        # Calculate total time
        total_time = time.time() - total_start_time
        avg_page_time = sum(page_times) / len(page_times) if page_times else 0
        
        # Create summary
        summary = {
            "doc_id": doc_id,
            "source_pdf": str(pdf_path),
            "num_pages": num_pages,
            "total_items": len(all_items),
            "item_breakdown": {
                "<FORM>": sum(1 for item in all_items if item.get('tag') == '<FORM>'),
                "<SH>": sum(1 for item in all_items if item.get('tag') == '<SH>'),
                "<Q>": sum(1 for item in all_items if item.get('tag') == '<Q>'),
                "<INPUT>": sum(1 for item in all_items if item.get('tag') == '<INPUT>'),
                "input_types": {
                    "option": sum(1 for item in all_items if item.get('tag') == '<INPUT>' and item.get('input_type') == 'option'),
                    "text": sum(1 for item in all_items if item.get('tag') == '<INPUT>' and item.get('input_type') == 'text'),
                    "free": sum(1 for item in all_items if item.get('tag') == '<INPUT>' and item.get('input_type') == 'free')
                }
            },
            "page_files": [str(p) for p in page_results],
            "processing_time": {
                "total_seconds": total_time,
                "avg_page_seconds": avg_page_time,
                "pages_per_minute": 60 / avg_page_time if avg_page_time > 0 else 0
            },
            "quality_control": {
                "avg_questions_per_page": sum(question_counts) / len(question_counts) if question_counts else 0,
                "min_questions_page": min(question_counts) if question_counts else 0,
                "max_questions_page": max(question_counts) if question_counts else 0,
                "pages_reexamined": len(pages_to_reexamine) if 'pages_to_reexamine' in locals() else 0,
                "pages_marked_unusual": len(pages_marked_unusual),
                "second_inspection_required": len(second_inspection_queue) > 0,
                "second_inspection_queue_file": str(doc_output_dir / "second_inspection_queue.json") if second_inspection_queue else None,
                "anomaly_detection": {
                    "statistical_threshold": "2_std_dev",
                    "ratio_thresholds": {
                        "excessive_questions": 2.0,
                        "too_few_questions": 0.3,
                        "high_supplemental_ratio": 0.9,
                        "low_supplemental_ratio": 0.1
                    }
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Save all items as JSON Lines
        jsonl_path = doc_output_dir / "all_items.jsonl"
        with open(jsonl_path, 'w') as f:
            for item in all_items:
                f.write(json.dumps(item) + '\n')
        
        # Save summary
        summary_path = doc_output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Digitization complete. Output: {doc_output_dir}")
        logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"Average per page: {avg_page_time:.2f} seconds")
        logger.info(f"Processing speed: {60 / avg_page_time if avg_page_time > 0 else 0:.1f} pages/minute")
        return summary


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage A: CRF Digitizer")
    parser.add_argument("pdf_path", type=str, help="Path to CRF PDF")
    parser.add_argument("--doc-id", type=str, help="Document ID (default: filename)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="Model path")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cuda, cpu)")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization")
    parser.add_argument("--fallback-only", action="store_true",
                       help="Use only fallback extraction (no vision model)")
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
        args.no_4bit = True  # Disable quantization on CPU
    
    try:
        # Initialize digitizer
        digitizer = CRFDigitizer(
            model_path=args.model,
            device=args.device,
            use_4bit=not args.no_4bit,
            fallback_only=args.fallback_only
        )
        
        # Process PDF
        result = digitizer.process_pdf(
            pdf_path=args.pdf_path,
            doc_id=args.doc_id
        )
        
        print(f"\nDigitization Summary:")
        print(f"- Document: {result['doc_id']}")
        print(f"- Pages: {result['num_pages']}")
        print(f"- Total items: {result['total_items']}")
        print(f"- Item breakdown: {result['item_breakdown']}")
        print(f"- Output: {Path(result['page_files'][0]).parent}")
        print(f"\nTiming:")
        print(f"- Total time: {result['processing_time']['total_seconds']:.2f} seconds")
        print(f"- Average per page: {result['processing_time']['avg_page_seconds']:.2f} seconds")
        print(f"- Processing speed: {result['processing_time']['pages_per_minute']:.1f} pages/minute")
        
    except Exception as e:
        logger.error(f"Error during digitization: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    main()