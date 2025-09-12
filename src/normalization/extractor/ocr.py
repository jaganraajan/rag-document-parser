"""
OCR Extractor - Basic Tesseract OCR Placeholder.

This module provides a placeholder for OCR functionality using Tesseract.
In a production environment, you would install pytesseract and tesseract-ocr.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import io

from src.normalization.schema import Section, BoundingBox

logger = logging.getLogger(__name__)


class OCRExtractor:
    """OCR extractor using Tesseract (placeholder implementation)."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ocr_available = self._check_ocr_availability()
    
    def _check_ocr_availability(self) -> bool:
        """Check if OCR dependencies are available."""
        try:
            import pytesseract
            return True
        except ImportError:
            self.logger.warning("pytesseract not available. OCR functionality will be limited.")
            return False
    
    def extract_text_from_image(self, image_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract text and layout from image using OCR.
        
        Returns:
            Tuple of (full_text, layout_elements, metadata)
        """
        if not self.ocr_available:
            return self._extract_dummy_content(image_path)
        
        try:
            # In production, this would use pytesseract
            return self._extract_with_tesseract(image_path)
        except Exception as e:
            self.logger.error(f"Error extracting from image {image_path}: {e}")
            return self._extract_dummy_content(image_path)
    
    def _extract_with_tesseract(self, image_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Extract using Tesseract OCR (requires pytesseract)."""
        try:
            import pytesseract
            from PIL import Image
            
            # Open image
            image = Image.open(image_path)
            
            # Get basic text
            text = pytesseract.image_to_string(image)
            
            # Get detailed data with bounding boxes
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Convert to layout elements
            layout_elements = self._convert_tesseract_to_layout(data, image.size)
            
            # Metadata
            metadata = {
                "ocr_engine": "tesseract",
                "image_width": image.size[0],
                "image_height": image.size[1],
                "confidence_mean": sum(data['conf']) / len(data['conf']) if data['conf'] else 0
            }
            
            return text, layout_elements, metadata
            
        except ImportError:
            self.logger.error("pytesseract not installed")
            return self._extract_dummy_content(image_path)
    
    def _convert_tesseract_to_layout(self, data: Dict[str, List], image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Convert Tesseract output to layout elements."""
        layout_elements = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text:  # Only include non-empty text
                left = data['left'][i]
                top = data['top'][i]
                width = data['width'][i]
                height = data['height'][i]
                
                bbox = (left, top, left + width, top + height)
                
                element = {
                    "content": text,
                    "type": "text",
                    "bbox": bbox,
                    "page": 1,  # Images are single page
                    "page_width": float(image_size[0]),
                    "page_height": float(image_size[1]),
                    "confidence": data['conf'][i] if data['conf'][i] > 0 else 50
                }
                layout_elements.append(element)
        
        return layout_elements
    
    def _extract_dummy_content(self, image_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Extract dummy content when OCR is not available."""
        try:
            # Try to open image to get dimensions
            image = Image.open(image_path)
            width, height = image.size
        except Exception:
            width, height = 800, 600  # Default dimensions
        
        # Create dummy content
        text = "[OCR_PLACEHOLDER] This is placeholder text from image OCR extraction. " \
               "To enable real OCR, install pytesseract: pip install pytesseract"
        
        layout_elements = [
            {
                "content": text,
                "type": "text",
                "bbox": (50, 50, width - 50, 100),
                "page": 1,
                "page_width": float(width),
                "page_height": float(height),
                "confidence": 0.1
            }
        ]
        
        metadata = {
            "ocr_engine": "placeholder",
            "image_width": width,
            "image_height": height,
            "note": "Real OCR not available - install pytesseract for actual text extraction"
        }
        
        return text, layout_elements, metadata
    
    def extract_from_pdf_scan(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract text from scanned PDF using OCR.
        This would typically convert PDF pages to images first.
        """
        if not self.ocr_available:
            return self._extract_dummy_pdf_scan(pdf_path)
        
        try:
            # In production, you would:
            # 1. Convert PDF pages to images using pdf2image
            # 2. Run OCR on each image
            # 3. Combine results
            return self._extract_dummy_pdf_scan(pdf_path)
        except Exception as e:
            self.logger.error(f"Error extracting from scanned PDF {pdf_path}: {e}")
            return self._extract_dummy_pdf_scan(pdf_path)
    
    def _extract_dummy_pdf_scan(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Dummy content for scanned PDF extraction."""
        text = "[PDF_SCAN_PLACEHOLDER] This is placeholder text from scanned PDF OCR extraction. " \
               "To enable real OCR, install pytesseract and pdf2image."
        
        layout_elements = [
            {
                "content": text,
                "type": "text",
                "bbox": (50, 50, 550, 100),
                "page": 1,
                "page_width": 612.0,
                "page_height": 792.0,
                "confidence": 0.1
            }
        ]
        
        metadata = {
            "ocr_engine": "placeholder",
            "scan_type": "pdf",
            "note": "Real OCR not available - install pytesseract and pdf2image"
        }
        
        return text, layout_elements, metadata
    
    def convert_to_sections(self, layout_elements: List[Dict[str, Any]]) -> List[Section]:
        """Convert OCR layout elements to document sections."""
        sections = []
        
        if not layout_elements:
            return sections
        
        # Group text by proximity and create sections
        # This is a simplified approach - in production you'd use more sophisticated grouping
        current_content = []
        
        for element in layout_elements:
            if element.get("type") == "text":
                current_content.append(element["content"])
        
        if current_content:
            # Create a single section for now
            bbox = None
            if layout_elements:
                first_elem = layout_elements[0]
                bbox = BoundingBox(
                    x1=first_elem["bbox"][0],
                    y1=first_elem["bbox"][1],
                    x2=first_elem["bbox"][2],
                    y2=first_elem["bbox"][3],
                    page=first_elem.get("page", 1)
                )
            
            section = Section(
                title="OCR Extracted Content",
                content="\n".join(current_content),
                level=1,
                bbox=bbox
            )
            sections.append(section)
        
        return sections
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats."""
        if self.ocr_available:
            return ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']
        else:
            return ['png', 'jpg', 'jpeg']  # PIL supports these even without OCR
    
    def is_image_file(self, filename: str) -> bool:
        """Check if file is a supported image format."""
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        return extension in self.get_supported_formats()


def install_ocr_dependencies():
    """Helper function to guide OCR installation."""
    instructions = """
    To enable OCR functionality, install the following:
    
    1. Install Tesseract OCR:
       - Ubuntu/Debian: sudo apt-get install tesseract-ocr
       - macOS: brew install tesseract
       - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
    
    2. Install Python packages:
       pip install pytesseract pdf2image
    
    3. For PDF scanning, also install:
       - Ubuntu/Debian: sudo apt-get install poppler-utils
       - macOS: brew install poppler
       - Windows: Download poppler and add to PATH
    """
    print(instructions)
    return instructions