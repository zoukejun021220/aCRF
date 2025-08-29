"""Fallback extraction for when vision model is not available"""

import logging
from typing import List, Optional

from ..models.data_classes import ExtractedLine, CRFItem

logger = logging.getLogger(__name__)


class FallbackExtractor:
    """Conservative fallback extraction - only extract clear questions"""
    
    @staticmethod
    def extract(lines: List[ExtractedLine], page_num: int, 
                current_section: Optional[str] = None) -> List[CRFItem]:
        """
        Conservative fallback extraction - only extract clear questions
        """
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