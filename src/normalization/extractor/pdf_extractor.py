"""
PDF Extractor with Layout Extraction.

This module handles PDF text and layout extraction using pdfplumber
for better layout analysis than the existing PyPDF2-based extractor.
"""

import pdfplumber
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

from src.normalization.schema import Section, BoundingBox


logger = logging.getLogger(__name__)


@dataclass
class PDFLayoutElement:
    """Represents a layout element extracted from PDF."""
    content: str
    element_type: str  # "text", "image", "table", "line"
    bbox: Tuple[float, float, float, float]
    page: int
    page_width: float
    page_height: float
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PDFExtractor:
    """Enhanced PDF extractor using pdfplumber for layout analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text_and_layout(self, file_path: str) -> Tuple[str, List[PDFLayoutElement], Dict[str, Any]]:
        """
        Extract text, layout elements, and metadata from PDF.
        
        Returns:
            Tuple of (full_text, layout_elements, metadata)
        """
        layout_elements = []
        full_text = ""
        metadata = {}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # Extract PDF metadata
                metadata = self._extract_metadata(pdf)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Get page dimensions
                    page_width = float(page.width)
                    page_height = float(page.height)
                    
                    # Extract text elements with positioning
                    text_elements = self._extract_text_elements(page, page_num, page_width, page_height)
                    layout_elements.extend(text_elements)
                    
                    # Extract tables
                    table_elements = self._extract_tables(page, page_num, page_width, page_height)
                    layout_elements.extend(table_elements)
                    
                    # Extract images/figures
                    image_elements = self._extract_images(page, page_num, page_width, page_height)
                    layout_elements.extend(image_elements)
                    
                    # Extract lines/drawings
                    line_elements = self._extract_lines(page, page_num, page_width, page_height)
                    layout_elements.extend(line_elements)
                    
                    # Add page text to full text
                    page_text = page.extract_text() or ""
                    full_text += page_text + "\n"
                
                metadata['page_count'] = len(pdf.pages)
                
        except Exception as e:
            self.logger.error(f"Error extracting PDF {file_path}: {e}")
            raise
        
        return full_text, layout_elements, metadata
    
    def _extract_metadata(self, pdf) -> Dict[str, Any]:
        """Extract PDF metadata."""
        metadata = {}
        
        if hasattr(pdf, 'metadata') and pdf.metadata:
            for key, value in pdf.metadata.items():
                if value is not None:
                    metadata[key.lower()] = str(value)
        
        return metadata
    
    def _extract_text_elements(self, page, page_num: int, page_width: float, page_height: float) -> List[PDFLayoutElement]:
        """Extract text elements with positioning information."""
        elements = []
        
        try:
            # Get characters with positioning
            chars = page.chars
            
            if not chars:
                return elements
            
            # Group characters into words and lines
            current_line = []
            current_bbox = None
            
            for char in chars:
                char_bbox = (char['x0'], char['top'], char['x1'], char['bottom'])
                
                # Check if this character continues the current line
                if current_line and self._is_same_line(current_bbox, char_bbox):
                    current_line.append(char)
                    current_bbox = self._merge_bboxes(current_bbox, char_bbox)
                else:
                    # Process previous line
                    if current_line:
                        line_element = self._create_text_element(
                            current_line, current_bbox, page_num, page_width, page_height
                        )
                        if line_element:
                            elements.append(line_element)
                    
                    # Start new line
                    current_line = [char]
                    current_bbox = char_bbox
            
            # Process last line
            if current_line:
                line_element = self._create_text_element(
                    current_line, current_bbox, page_num, page_width, page_height
                )
                if line_element:
                    elements.append(line_element)
                    
        except Exception as e:
            self.logger.warning(f"Error extracting text elements from page {page_num}: {e}")
        
        return elements
    
    def _is_same_line(self, bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float], tolerance: float = 5.0) -> bool:
        """Check if two bounding boxes are on the same line."""
        if bbox1 is None or bbox2 is None:
            return False
        
        # Check if vertical positions overlap
        return abs(bbox1[1] - bbox2[1]) <= tolerance and abs(bbox1[3] - bbox2[3]) <= tolerance
    
    def _merge_bboxes(self, bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Merge two bounding boxes."""
        return (
            min(bbox1[0], bbox2[0]),  # x0
            min(bbox1[1], bbox2[1]),  # y0
            max(bbox1[2], bbox2[2]),  # x1
            max(bbox1[3], bbox2[3])   # y1
        )
    
    def _create_text_element(self, chars: List[Dict], bbox: Tuple[float, float, float, float],
                           page_num: int, page_width: float, page_height: float) -> Optional[PDFLayoutElement]:
        """Create a text element from a list of characters."""
        if not chars:
            return None
        
        # Extract text content
        content = ''.join(char.get('text', '') for char in chars)
        content = content.strip()
        
        if not content:
            return None
        
        # Get font information from first character
        first_char = chars[0]
        font_size = first_char.get('size')
        font_name = first_char.get('fontname')
        
        return PDFLayoutElement(
            content=content,
            element_type="text",
            bbox=bbox,
            page=page_num,
            page_width=page_width,
            page_height=page_height,
            font_size=font_size,
            font_name=font_name,
            metadata={"char_count": len(chars)}
        )
    
    def _extract_tables(self, page, page_num: int, page_width: float, page_height: float) -> List[PDFLayoutElement]:
        """Extract table elements."""
        elements = []
        
        try:
            tables = page.find_tables()
            
            for i, table in enumerate(tables):
                if table.bbox:
                    # Convert table to text representation
                    table_data = table.extract()
                    if table_data:
                        content = self._table_to_text(table_data)
                        
                        element = PDFLayoutElement(
                            content=content,
                            element_type="table",
                            bbox=table.bbox,
                            page=page_num,
                            page_width=page_width,
                            page_height=page_height,
                            metadata={
                                "table_index": i,
                                "rows": len(table_data),
                                "cols": len(table_data[0]) if table_data else 0
                            }
                        )
                        elements.append(element)
                        
        except Exception as e:
            self.logger.warning(f"Error extracting tables from page {page_num}: {e}")
        
        return elements
    
    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert table data to text representation."""
        lines = []
        for row in table_data:
            if row:
                # Clean and join cells
                clean_row = [str(cell or '').strip() for cell in row]
                lines.append(' | '.join(clean_row))
        return '\n'.join(lines)
    
    def _extract_images(self, page, page_num: int, page_width: float, page_height: float) -> List[PDFLayoutElement]:
        """Extract image/figure elements."""
        elements = []
        
        try:
            # pdfplumber doesn't directly extract images, but we can detect image areas
            # This is a simplified approach - in production you might use other libraries
            images = getattr(page, 'images', [])
            
            for i, image in enumerate(images):
                bbox = (image['x0'], image['top'], image['x1'], image['bottom'])
                
                element = PDFLayoutElement(
                    content=f"[IMAGE_{i}]",
                    element_type="image",
                    bbox=bbox,
                    page=page_num,
                    page_width=page_width,
                    page_height=page_height,
                    metadata={"image_index": i}
                )
                elements.append(element)
                
        except Exception as e:
            self.logger.warning(f"Error extracting images from page {page_num}: {e}")
        
        return elements
    
    def _extract_lines(self, page, page_num: int, page_width: float, page_height: float) -> List[PDFLayoutElement]:
        """Extract line/drawing elements."""
        elements = []
        
        try:
            lines = getattr(page, 'lines', [])
            
            for i, line in enumerate(lines):
                if 'x0' in line and 'top' in line:
                    bbox = (line['x0'], line['top'], line.get('x1', line['x0']), line.get('bottom', line['top']))
                    
                    element = PDFLayoutElement(
                        content=f"[LINE_{i}]",
                        element_type="line",
                        bbox=bbox,
                        page=page_num,
                        page_width=page_width,
                        page_height=page_height,
                        metadata={"line_index": i}
                    )
                    elements.append(element)
                    
        except Exception as e:
            self.logger.warning(f"Error extracting lines from page {page_num}: {e}")
        
        return elements
    
    def convert_to_sections(self, layout_elements: List[PDFLayoutElement]) -> List[Section]:
        """Convert layout elements to document sections."""
        sections = []
        
        # Group text elements by page and proximity
        text_elements = [e for e in layout_elements if e.element_type == "text"]
        
        if not text_elements:
            return sections
        
        # Sort by page, then by y-coordinate (top to bottom)
        text_elements.sort(key=lambda e: (e.page, e.bbox[1]))
        
        current_section = None
        
        for element in text_elements:
            # Check if this looks like a header (larger font, short text)
            is_header = (
                element.font_size and element.font_size > 12 and 
                len(element.content) < 100 and 
                not element.content.endswith('.')
            )
            
            if is_header:
                # Start new section
                if current_section:
                    sections.append(current_section)
                
                bbox = BoundingBox(
                    x1=element.bbox[0], y1=element.bbox[1],
                    x2=element.bbox[2], y2=element.bbox[3],
                    page=element.page
                )
                
                current_section = Section(
                    title=element.content,
                    content="",
                    level=1,
                    bbox=bbox
                )
            else:
                # Add to current section or create default section
                if current_section:
                    current_section.content += element.content + "\n"
                else:
                    # Create default section for content without header
                    bbox = BoundingBox(
                        x1=element.bbox[0], y1=element.bbox[1],
                        x2=element.bbox[2], y2=element.bbox[3],
                        page=element.page
                    )
                    
                    current_section = Section(
                        title="Content",
                        content=element.content + "\n",
                        level=1,
                        bbox=bbox
                    )
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def get_layout_elements_dict(self, layout_elements: List[PDFLayoutElement]) -> List[Dict[str, Any]]:
        """Convert layout elements to dictionary format for signature generation."""
        return [
            {
                "content": element.content,
                "type": element.element_type,
                "bbox": element.bbox,
                "page": element.page,
                "page_width": element.page_width,
                "page_height": element.page_height,
                "font_size": element.font_size,
                "font_name": element.font_name
            }
            for element in layout_elements
        ]