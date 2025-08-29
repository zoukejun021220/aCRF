"""Data classes for Stage A CRF Digitizer"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

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
    def from_pdf_rect(cls, rect, page_width: float, page_height: float) -> 'BoundingBox':
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