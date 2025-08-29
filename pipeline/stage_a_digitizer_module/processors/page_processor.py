"""Page processing logic for CRF extraction"""

import logging
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
from PIL import Image

from ..models.data_classes import ExtractedLine
from ..processors.extraction_prompts import (
    QUESTION_EXTRACTION_PROMPT,
    SECTION_EXTRACTION_PROMPT,
    FORM_NAME_EXTRACTION_PROMPT
)
from ..processors.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class PageProcessor:
    """Handles page-level processing logic"""
    
    def __init__(self, vision_model, current_seq: int = 1, global_question_counter: int = 0):
        self.vision_model = vision_model
        self.current_seq = current_seq
        self.global_question_counter = global_question_counter
        self.parser = ResponseParser()
    
    def process_page(self, page_image: Image.Image, lines: List[ExtractedLine], 
                     page_num: int, current_form: Optional[str] = None, 
                     current_section: Optional[str] = None) -> Tuple[List[Dict], Optional[str], Optional[str]]:
        """
        Process a single page using three-pass extraction
        
        Returns:
            Tuple of (extracted items, updated form name, updated section)
        """
        # Step 1: Extract question-input pairs
        logger.info(f"[Page {page_num + 1}] Step 1/3: Extracting question-input pairs...")
        question_items = self._extract_questions(page_image, page_num)
        
        # Step 2: Extract form name
        logger.info(f"[Page {page_num + 1}] Step 2/3: Extracting form name...")
        form_items = self._extract_form_name(page_image, page_num)
        
        # Step 3: Extract section headers
        logger.info(f"[Page {page_num + 1}] Step 3/3: Extracting section headers...")
        section_items = self._extract_sections(page_image, page_num, question_items, form_items)
        
        # Clean sections that are too similar to questions
        logger.info(f"[Page {page_num + 1}] Cleaning sections similar to questions...")
        section_items = self._clean_similar_sections(section_items, question_items)
        
        # Process and combine all items
        all_items = []
        
        # Update current form
        for item in form_items:
            if item.get('text'):
                current_form = item['text']
                all_items.append(item)
        
        # Add section headers
        if section_items:
            all_items.extend(section_items)
            # Update current section to last found
            last_section = section_items[-1]
            if last_section.get('text'):
                current_section = last_section['text']
        
        # Add questions with temporary section assignment
        for item in question_items:
            if item.get('tag') == '<Q>':
                item['form'] = current_form
                item['section'] = current_section
            elif item.get('tag') == '<INPUT>':
                item['form'] = current_form
                item['section'] = current_section
            all_items.append(item)
        
        # Match items with bounding boxes
        matched_items = self._match_items_to_lines(all_items, lines)
        
        # Assign proper question IDs
        matched_items = self._assign_question_ids(matched_items)
        
        # Assign sections based on question_below_text
        matched_items = self._assign_sections_by_question_below(matched_items, current_section)
        
        # Update current_section from processed items
        for item in reversed(matched_items):
            if item.get('tag') == '<SH>' and item.get('text'):
                current_section = item['text']
                break
        
        # Update sequence counters
        self.current_seq = max(self.current_seq, max((item.get('seq', 0) for item in matched_items), default=0) + 1)
        
        return matched_items, current_form, current_section
    
    def _extract_questions(self, page_image: Image.Image, page_num: int) -> List[Dict]:
        """Extract question-input pairs"""
        try:
            response = self.vision_model.query_model(page_image, QUESTION_EXTRACTION_PROMPT)
            logger.debug(f"Raw model output (questions): {response}")
            items = self.parser.parse_simple_tags(
                response, page_num, 'questions', self.current_seq
            )
            # Update current_seq
            if items:
                self.current_seq = max(self.current_seq, max(item.get('seq', 0) for item in items) + 1)
            return items
        except Exception as e:
            logger.error(f"Error extracting questions: {e}")
            return []
    
    def _extract_form_name(self, page_image: Image.Image, page_num: int) -> List[Dict]:
        """Extract form name"""
        try:
            response = self.vision_model.query_model(page_image, FORM_NAME_EXTRACTION_PROMPT)
            logger.debug(f"Raw model output (form): {response}")
            items = self.parser.parse_simple_tags(
                response, page_num, 'form', self.current_seq
            )
            if items:
                self.current_seq = max(self.current_seq, max(item.get('seq', 0) for item in items) + 1)
            return items
        except Exception as e:
            logger.error(f"Error extracting form name: {e}")
            return []
    
    def _extract_sections(self, page_image: Image.Image, page_num: int, 
                         question_items: List[Dict], form_items: List[Dict]) -> List[Dict]:
        """Extract section headers"""
        # Build exclusion list
        extracted_questions = []
        for item in question_items:
            if item.get('tag') == '<Q>' and item.get('text'):
                extracted_questions.append(item['text'])
        
        questions_list = "\n".join(f"- {q}" for q in extracted_questions[:10])
        if len(extracted_questions) > 10:
            questions_list += f"\n... and {len(extracted_questions) - 10} more questions"
        
        form_name = None
        for item in form_items:
            if item.get('tag') == '<FORM>' and item.get('text'):
                form_name = item['text']
                break
        
        prompt = SECTION_EXTRACTION_PROMPT.format(
            form_name=form_name if form_name else "(none found)",
            questions_list=questions_list if extracted_questions else "(none found)"
        )
        
        try:
            response = self.vision_model.query_model(page_image, prompt)
            logger.debug(f"Raw model output (sections): {response}")
            items = self.parser.parse_simple_tags(
                response, page_num, 'sections', self.current_seq
            )
            if items:
                self.current_seq = max(self.current_seq, max(item.get('seq', 0) for item in items) + 1)
            return items
        except Exception as e:
            logger.error(f"Error extracting sections: {e}")
            return []
    
    def _clean_similar_sections(self, section_items: List[Dict], question_items: List[Dict], 
                               similarity_threshold: float = 0.9) -> List[Dict]:
        """Remove section headers too similar to questions"""
        cleaned_sections = []
        
        # Extract texts for comparison
        question_texts = [q.get('text', '').strip().lower() 
                         for q in question_items 
                         if q.get('tag') == '<Q>' and q.get('text')]
        
        input_texts = [i.get('text', '').strip().lower() 
                      for i in question_items 
                      if i.get('tag') == '<INPUT>' and i.get('text')]
        
        all_texts_to_check = question_texts + input_texts
        
        for section in section_items:
            if section.get('tag') != '<SH>' or not section.get('text'):
                cleaned_sections.append(section)
                continue
                
            section_text = section['text'].strip().lower()
            is_too_similar = False
            
            for text in all_texts_to_check:
                similarity = SequenceMatcher(None, section_text, text).ratio()
                if similarity >= similarity_threshold:
                    logger.debug(f"Removing section '{section['text']}' - too similar to '{text}'")
                    is_too_similar = True
                    break
            
            if not is_too_similar:
                cleaned_sections.append(section)
                
        return cleaned_sections
    
    def _match_items_to_lines(self, items: List[Dict], lines: List[ExtractedLine]) -> List[Dict]:
        """Match extracted items to OCR lines for bounding boxes"""
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
    
    def _assign_question_ids(self, items: List[Dict]) -> List[Dict]:
        """Assign proper question IDs based on sequence order"""
        # Separate by type
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
        
        # Sort questions by sequence number
        questions.sort(key=lambda x: x.get('seq', 0))
        
        # Assign IDs and build mapping
        temp_to_final_qid = {}
        for question in questions:
            final_qid = f"q-{self.global_question_counter:03d}"
            temp_qid = question.get('temp_qid')
            if temp_qid:
                temp_to_final_qid[temp_qid] = final_qid
                del question['temp_qid']
            question['qid'] = final_qid
            logger.debug(f"Assigned {final_qid} to: {question['text'][:50]}...")
            self.global_question_counter += 1
        
        # Update parent_qid for inputs
        inputs.sort(key=lambda x: (x['bbox_xyxy'][1], x['bbox_xyxy'][0]))
        
        for input_item in inputs:
            parent_temp_qid = input_item.get('parent_temp_qid')
            
            if parent_temp_qid and parent_temp_qid in temp_to_final_qid:
                input_item['parent_qid'] = temp_to_final_qid[parent_temp_qid]
                if 'parent_temp_qid' in input_item:
                    del input_item['parent_temp_qid']
            else:
                # Find closest question above
                input_y = input_item['bbox_xyxy'][1]
                input_x = input_item['bbox_xyxy'][0]
                
                best_qid = None
                best_score = float('inf')
                
                for question in questions:
                    q_y = question['bbox_xyxy'][1]
                    q_x = question['bbox_xyxy'][0]
                    
                    if q_y <= input_y:
                        y_distance = input_y - q_y
                        x_distance = abs(input_x - q_x)
                        score = y_distance + (x_distance * 0.1 if x_distance < 200 else x_distance)
                        
                        if input_item.get('input_type') == 'option' and x_distance < 100:
                            score *= 0.8
                        
                        if score < best_score:
                            best_score = score
                            best_qid = question['qid']
                
                if best_qid and best_score < 500:
                    input_item['parent_qid'] = best_qid
                else:
                    input_item['parent_qid'] = 'orphan'
                
                if 'parent_temp_qid' in input_item:
                    del input_item['parent_temp_qid']
        
        # Combine and sort
        all_items = questions + inputs + other_items
        all_items.sort(key=lambda x: x.get('seq', 0))
        
        return all_items
    
    def _assign_sections_by_question_below(self, items: List[Dict], 
                                          current_section: Optional[str] = None) -> List[Dict]:
        """Assign sections to questions based on question_below_text"""
        # Find section headers and their ranges
        sections = [item for item in items if item.get('tag') == '<SH>']
        sections.sort(key=lambda x: x.get('seq', 0))
        
        # Create section ranges
        section_ranges = []
        for i, section in enumerate(sections):
            section_name = section['text']
            section_seq = section['seq']
            question_below_text = section.get('question_below_text', 'none')
            
            # Determine section start
            if question_below_text != 'none':
                start_seq = None
                for item in items:
                    if item.get('tag') == '<Q>' and item.get('seq', 0) > section_seq:
                        if item['text'] == question_below_text or question_below_text in item['text']:
                            start_seq = item['seq']
                            break
                if start_seq is None:
                    start_seq = section_seq + 0.1
            else:
                start_seq = section_seq + 0.1
            
            # Determine section end
            if i + 1 < len(sections):
                end_seq = sections[i + 1]['seq']
            else:
                end_seq = float('inf')
            
            section_ranges.append({
                'name': section_name,
                'section_seq': section_seq,
                'start_seq': start_seq,
                'end_seq': end_seq
            })
        
        # Assign sections to items
        for item in items:
            if item.get('tag') == '<Q>':
                item_seq = item.get('seq', 0)
                assigned_section = current_section
                
                for section_range in section_ranges:
                    if section_range['section_seq'] < item_seq < section_range['end_seq']:
                        if item_seq >= section_range['start_seq']:
                            assigned_section = section_range['name']
                            break
                
                item['section'] = assigned_section
                
            elif item.get('tag') == '<INPUT>':
                parent_qid = item.get('parent_qid')
                if parent_qid:
                    for q_item in items:
                        if q_item.get('tag') == '<Q>' and q_item.get('qid') == parent_qid:
                            item['section'] = q_item.get('section', current_section)
                            break
                else:
                    item['section'] = current_section
        
        return items