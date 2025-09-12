"""
Document Pipeline Orchestrator.

This module orchestrates the end-to-end document normalization pipeline,
coordinating signature matching, rule application, LLM fallback, and storage.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from src.normalization.schema import NormalizedDocument, calculate_coverage_stats
from src.normalization.signatures import SignatureManager
from src.normalization.rules_engine import RulesEngine
from src.normalization.extractor.pdf_extractor import PDFExtractor
from src.normalization.extractor.ocr import OCRExtractor
from src.normalization.extractor.email_extractor import EmailExtractor
from src.normalization.model_fallback.llm_field_extractor import LLMFieldExtractor, create_provider
from src.normalization.logs.interpretation import InterpretationLogger
from src.normalization.storage.repository import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Main document processing pipeline."""
    
    def __init__(self, 
                 rules_dir: str = "rules",
                 signatures_dir: str = "signatures", 
                 outputs_dir: str = "outputs",
                 logs_dir: str = "logs",
                 llm_provider: str = "dummy",
                 llm_config: Dict[str, Any] = None):
        """
        Initialize the document pipeline.
        
        Args:
            rules_dir: Directory containing rule files
            signatures_dir: Directory for signature storage
            outputs_dir: Directory for normalized document outputs
            logs_dir: Directory for interpretation logs
            llm_provider: LLM provider type ("dummy", "openai", "gemini")
            llm_config: Configuration for LLM provider
        """
        self.rules_dir = rules_dir
        self.signatures_dir = signatures_dir
        self.outputs_dir = outputs_dir
        self.logs_dir = logs_dir
        
        # Initialize components
        self.signature_manager = SignatureManager(signatures_dir)
        self.rules_engine = RulesEngine(rules_dir)
        self.interpretation_logger = InterpretationLogger(logs_dir)
        self.repository = DocumentRepository(outputs_dir)
        
        # Initialize extractors
        self.pdf_extractor = PDFExtractor()
        self.ocr_extractor = OCRExtractor()
        self.email_extractor = EmailExtractor()
        
        # Initialize LLM fallback
        llm_config = llm_config or {}
        provider = create_provider(llm_provider, **llm_config)
        self.llm_extractor = LLMFieldExtractor(provider)
        
        logger.info(f"Pipeline initialized with {llm_provider} LLM provider")
    
    def process_document(self, file_path: str, max_cost_usd: float = 1.0) -> Tuple[NormalizedDocument, str]:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            max_cost_usd: Maximum cost limit for LLM calls
            
        Returns:
            Tuple of (normalized_document, log_file_path)
        """
        start_time = time.time()
        
        # Read file
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        file_size = len(file_content)
        filename = os.path.basename(file_path)
        file_type = self._detect_file_type(filename)
        
        # Create normalized document
        document = NormalizedDocument.create(filename, file_size, file_type, file_content)
        
        # Start logging
        log_file = self.interpretation_logger.start_document_logging(document.doc_id)
        
        try:
            # Stage 1: INGEST
            self.interpretation_logger.log_ingest(filename, file_size, file_type)
            text, layout_elements = self._extract_content(file_path, file_type)
            
            # Stage 2: SIGNATURE
            signature, similarity_score = self._process_signature(layout_elements, filename)
            document.processing_meta.signature_id = signature.signature_id
            document.processing_meta.signature_match_score = similarity_score
            
            # Stage 3: RULES
            extracted_values, rules_applied = self._apply_rules(text, signature.signature_id)
            document.key_values.extend(extracted_values)
            document.processing_meta.rules_applied = rules_applied
            
            # Check coverage
            required_fields = self.rules_engine.get_required_fields(signature.signature_id)
            missing_fields = self._find_missing_fields(extracted_values, required_fields)
            
            # Stage 4: LLM (if needed)
            if missing_fields:
                self.interpretation_logger.log_missing_fields(missing_fields, True)
                llm_extracted = self._apply_llm_fallback(missing_fields, text, max_cost_usd)
                document.key_values.extend(llm_extracted)
                document.processing_meta.model_calls_made = len(llm_extracted)
                document.processing_meta.total_cost_usd = self.llm_extractor.total_cost_usd
            else:
                self.interpretation_logger.log_missing_fields([], False)
            
            # Stage 5: FINALIZE
            self._finalize_document(document, text, layout_elements)
            
            # Calculate final stats
            processing_time = time.time() - start_time
            document.ingest_metadata.processing_time_seconds = processing_time
            coverage_stats = calculate_coverage_stats(document)
            document.processing_meta.coverage_stats = coverage_stats
            
            # Save document
            save_path = self.repository.save_document(document)
            document.interpretation_log_path = log_file
            
            # Log completion
            self.interpretation_logger.log_processing_complete(
                document.processing_meta.total_cost_usd,
                processing_time,
                len(document.key_values),
                coverage_stats
            )
            
            logger.info(f"Processed document {document.doc_id} in {processing_time:.2f}s "
                       f"(cost: ${document.processing_meta.total_cost_usd:.4f})")
            
            return document, log_file
            
        except Exception as e:
            self.interpretation_logger.log_error("PIPELINE", "processing_error", str(e))
            logger.error(f"Error processing document: {e}")
            raise
        finally:
            self.interpretation_logger.finish_document_logging()
    
    def _detect_file_type(self, filename: str) -> str:
        """Detect file type from filename."""
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        type_mapping = {
            'pdf': 'pdf',
            'png': 'image',
            'jpg': 'image',
            'jpeg': 'image',
            'tiff': 'image',
            'bmp': 'image',
            'gif': 'image',
            'eml': 'email',
            'msg': 'email',
            'txt': 'text',
            'html': 'html',
            'htm': 'html'
        }
        
        return type_mapping.get(extension, 'unknown')
    
    def _extract_content(self, file_path: str, file_type: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text and layout elements based on file type."""
        if file_type == 'pdf':
            text, layout_elements, metadata = self.pdf_extractor.extract_text_and_layout(file_path)
            return text, self.pdf_extractor.get_layout_elements_dict(layout_elements)
        
        elif file_type == 'image':
            text, layout_elements, metadata = self.ocr_extractor.extract_text_from_image(file_path)
            return text, layout_elements
        
        elif file_type == 'email':
            text, layout_elements, metadata = self.email_extractor.extract_from_email(file_path)
            return text, layout_elements
        
        elif file_type in ['text', 'html']:
            # Simple text extraction
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Create simple layout element
            layout_elements = [{
                'content': text,
                'type': 'text',
                'bbox': (0, 0, 612, 792),
                'page': 1,
                'page_width': 612.0,
                'page_height': 792.0
            }]
            
            return text, layout_elements
        
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _process_signature(self, layout_elements: List[Dict[str, Any]], filename: str) -> Tuple[Any, float]:
        """Process layout signature matching."""
        signature, similarity_score = self.signature_manager.create_or_match_signature(
            layout_elements, filename
        )
        
        is_new = similarity_score == 1.0 and signature.document_count == 1
        self.interpretation_logger.log_signature_creation(
            signature.signature_id, is_new, similarity_score
        )
        
        return signature, similarity_score
    
    def _apply_rules(self, text: str, signature_id: str) -> Tuple[List[Any], List[str]]:
        """Apply rule-based extraction."""
        extracted_values, rules_applied = self.rules_engine.apply_rules(text, signature_id)
        
        # Log rule application
        self.interpretation_logger.log_rules_application(rules_applied, len(extracted_values))
        
        # Log individual field extractions
        for kv in extracted_values:
            self.interpretation_logger.log_field_extraction(
                kv.key, kv.value, kv.extraction_method, kv.confidence
            )
        
        return extracted_values, rules_applied
    
    def _find_missing_fields(self, extracted_values: List[Any], required_fields: List[str]) -> List[str]:
        """Find fields that are required but not extracted."""
        extracted_field_names = {kv.key for kv in extracted_values}
        missing_fields = [field for field in required_fields if field not in extracted_field_names]
        return missing_fields
    
    def _apply_llm_fallback(self, missing_fields: List[str], text: str, max_cost: float) -> List[Any]:
        """Apply LLM-based extraction for missing fields."""
        # Extract using LLM
        llm_extracted = self.llm_extractor.extract_missing_fields(
            missing_fields, [], text, max_cost
        )
        
        # Log LLM calls
        for call in self.llm_extractor.call_history:
            if not hasattr(call, 'logged'):  # Avoid double logging
                self.interpretation_logger.log_llm_invocation(
                    call.field_name, call.success, call.cost_usd, call.model_name
                )
                call.logged = True
        
        # Log individual extractions
        for kv in llm_extracted:
            self.interpretation_logger.log_field_extraction(
                kv.key, kv.value, kv.extraction_method, kv.confidence
            )
        
        return llm_extracted
    
    def _finalize_document(self, document: NormalizedDocument, text: str, layout_elements: List[Dict[str, Any]]):
        """Finalize document with sections and chunks."""
        # Create sections based on file type
        if document.ingest_metadata.file_type == 'pdf':
            # Convert layout elements back to PDFLayoutElement objects for section creation
            from src.normalization.extractor.pdf_extractor import PDFLayoutElement
            pdf_elements = [
                PDFLayoutElement(
                    content=elem['content'],
                    element_type=elem['type'],
                    bbox=elem['bbox'],
                    page=elem['page'],
                    page_width=elem['page_width'],
                    page_height=elem['page_height'],
                    font_size=elem.get('font_size'),
                    font_name=elem.get('font_name')
                )
                for elem in layout_elements
            ]
            document.sections = self.pdf_extractor.convert_to_sections(pdf_elements)
        
        elif document.ingest_metadata.file_type == 'email':
            document.sections = self.email_extractor.convert_to_sections(layout_elements)
        
        elif document.ingest_metadata.file_type == 'image':
            document.sections = self.ocr_extractor.convert_to_sections(layout_elements)
        
        else:
            # Create a simple section for text files
            from src.normalization.schema import Section
            document.sections = [Section(
                title="Document Content",
                content=text,
                level=1
            )]
        
        # Create chunks for RAG (simplified)
        document.chunks = self._create_chunks(text, document.doc_id)
        
        # Update page count
        document.ingest_metadata.page_count = max(
            elem.get('page', 1) for elem in layout_elements
        ) if layout_elements else 1
    
    def _create_chunks(self, text: str, doc_id: str) -> List[Any]:
        """Create document chunks for RAG."""
        from src.normalization.schema import Chunk
        
        # Simple chunking by paragraphs (in production, use more sophisticated chunking)
        paragraphs = text.split('\n\n')
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                chunk = Chunk(
                    content=paragraph.strip(),
                    chunk_id=f"{doc_id}_chunk_{i:03d}",
                    start_page=1,
                    end_page=1,
                    tokens=len(paragraph.split())
                )
                chunks.append(chunk)
        
        return chunks
    
    def process_batch(self, file_paths: List[str], max_cost_per_doc: float = 1.0) -> List[Tuple[str, Optional[str]]]:
        """
        Process multiple documents in batch.
        
        Returns:
            List of (doc_id, error_message) tuples
        """
        results = []
        
        for file_path in file_paths:
            try:
                document, log_file = self.process_document(file_path, max_cost_per_doc)
                results.append((document.doc_id, None))
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append((None, str(e)))
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return {
            "signatures": self.signature_manager.get_signature_stats(),
            "rules": self.rules_engine.get_stats(),
            "llm": self.llm_extractor.get_stats(),
            "repository": self.repository.get_statistics(),
            "costs": self.interpretation_logger.get_cost_summary()
        }
    
    def export_pipeline_config(self, output_path: str):
        """Export current pipeline configuration."""
        config = {
            "pipeline_version": "1.0.0",
            "directories": {
                "rules": self.rules_dir,
                "signatures": self.signatures_dir,
                "outputs": self.outputs_dir,
                "logs": self.logs_dir
            },
            "llm_provider": self.llm_extractor.provider.get_name(),
            "stats": self.get_pipeline_stats()
        }
        
        with open(output_path, 'w') as f:
            import json
            json.dump(config, f, indent=2, default=str)
    
    def clear_caches(self):
        """Clear all pipeline caches."""
        self.llm_extractor.clear_cache()
        self.signature_manager.load_signatures()  # Reload from disk
        self.rules_engine.load_rules()  # Reload rules