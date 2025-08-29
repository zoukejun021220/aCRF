"""PDF extraction utilities for Stage A"""

import fitz  # PyMuPDF
from PIL import Image
import io
from pathlib import Path
from typing import List, Tuple

from ..models.data_classes import ExtractedLine, BoundingBox


class PDFExtractor:
    """Handles PDF extraction operations"""
    
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
    
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
        mat = fitz.Matrix(self.dpi/72, self.dpi/72)
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