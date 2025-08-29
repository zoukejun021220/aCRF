"""Core CRF Digitizer class that orchestrates the extraction process"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import torch
import fitz  # PyMuPDF

from ..models.vision_model import VisionModel
from ..models.data_classes import CRFItem
from ..processors.page_processor import PageProcessor
from ..processors.fallback_extractor import FallbackExtractor
from ..utils.pdf_extractor import PDFExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        
        # Initialize components
        if not fallback_only:
            self.vision_model = VisionModel(model_path, device, use_4bit)
            self.vision_model.initialize()
        else:
            self.vision_model = None
            logger.info("Using fallback-only mode (no vision model)")
        
        self.pdf_extractor = PDFExtractor(dpi)
        
        # Output directories
        self.output_dir = Path("crf_json")
        self.output_dir.mkdir(exist_ok=True)
        
        # Track sequence number across pages
        self.current_seq = 1
        # Track question number across pages
        self.global_question_counter = 0
    
    def extract_page_primitives(self, pdf_path: Path, page_num: int):
        """Extract page image and text lines with bounding boxes"""
        return self.pdf_extractor.extract_page_primitives(pdf_path, page_num)
    
    def digitize_page(self, page_image, lines, page_num: int, 
                     current_form: Optional[str] = None, 
                     current_section: Optional[str] = None, 
                     max_retries: int = 3) -> Tuple[List[Dict], Optional[str], Optional[str]]:
        """
        Use Qwen-VL to extract CRF items from page using 3 separate queries
        
        Returns:
            Tuple of (List of extracted items as dicts, updated form name, updated section)
        """
        # Use fallback if in fallback-only mode
        if self.fallback_only:
            items = FallbackExtractor.extract(lines, page_num, current_section)
            # Convert CRFItem objects to dicts
            items_dict = [item.to_dict() for item in items]
            return items_dict, current_form, current_section
        
        # Clear memory cache for new page session
        if self.clear_cache_per_page and self.device == "cuda":
            self.vision_model.clear_cache()
        
        # Initialize page processor with current counters
        processor = PageProcessor(
            self.vision_model, 
            self.current_seq, 
            self.global_question_counter
        )
        
        # Process the page
        matched_items, current_form, current_section = processor.process_page(
            page_image, lines, page_num, current_form, current_section
        )
        
        # Update counters from processor
        self.current_seq = processor.current_seq
        self.global_question_counter = processor.global_question_counter
        
        # Clear memory cache after page processing
        if self.clear_cache_per_page and self.device == "cuda":
            self.vision_model.clear_cache()
        
        return matched_items, current_form, current_section
    
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
            items, current_form, current_section = self.digitize_page(
                page_image, lines, page_num, current_form, current_section
            )
            
            # Extract supplemental page data (non-question text)
            supplemental_data = []
            
            if items:
                # If we have questions, supplemental data is everything NOT in questions/inputs
                question_texts = {item.get('text', '').lower() 
                                for item in items 
                                if item.get('tag') in ['<Q>', '<INPUT>']}
                
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
                    questions = page_data.get('items', [])
                    malformed_count = 0
                    for q in questions:
                        if q.get('tag') == '<Q>':
                            # Check for overly long sections (likely parsing error)
                            if q.get('section') and len(q['section']) > 50:
                                malformed_count += 1
                            # Check for missing critical fields
                            if not q.get('text') or not q.get('bbox_xyxy'):
                                malformed_count += 1
                    
                    if questions and malformed_count > len([q for q in questions if q.get('tag') == '<Q>']) * 0.3:
                        reasons.append(f"high_malformed_questions ({malformed_count}/{len(questions)})")
                
                if reasons:
                    pages_marked_unusual.append(page_num)
                    anomaly_reasons[page_num] = reasons
                    logger.warning(f"Page {page_num + 1} marked as unusual: {', '.join(reasons)}")
        
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