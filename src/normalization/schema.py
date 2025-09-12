"""
Normalized JSON Schema for the hybrid document normalization pipeline.

This module defines the stable, extensible schema for capturing document 
ingest metadata, sections, key_values, tables, chunking, processing meta, 
and interpretation log reference.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
import hashlib


@dataclass
class BoundingBox:
    """Represents a bounding box for layout elements."""
    x1: float
    y1: float
    x2: float
    y2: float
    page: int = 1


@dataclass
class Section:
    """Represents a document section."""
    title: str
    content: str
    level: int = 1
    bbox: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyValue:
    """Represents a key-value pair extracted from the document."""
    key: str
    value: Union[str, float, int, bool]
    confidence: float = 1.0
    bbox: Optional[BoundingBox] = None
    extraction_method: str = "unknown"  # "rule", "model", "heuristic"


@dataclass
class TableCell:
    """Represents a single table cell."""
    content: str
    row: int
    col: int
    bbox: Optional[BoundingBox] = None


@dataclass
class Table:
    """Represents a table extracted from the document."""
    cells: List[TableCell]
    headers: List[str] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Represents a document chunk for RAG."""
    content: str
    chunk_id: str
    start_page: int
    end_page: int
    tokens: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestMetadata:
    """Metadata about the document ingestion process."""
    filename: str
    file_size: int
    file_type: str
    uploaded_at: datetime
    content_hash: str
    page_count: int = 0
    processing_time_seconds: float = 0.0


@dataclass
class ProcessingMeta:
    """Metadata about the processing pipeline."""
    pipeline_version: str = "1.0.0"
    signature_id: Optional[str] = None
    signature_match_score: float = 0.0
    rules_applied: List[str] = field(default_factory=list)
    model_calls_made: int = 0
    total_cost_usd: float = 0.0
    coverage_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedDocument:
    """The main normalized document schema."""
    doc_id: str
    ingest_metadata: IngestMetadata
    sections: List[Section] = field(default_factory=list)
    key_values: List[KeyValue] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    processing_meta: ProcessingMeta = field(default_factory=ProcessingMeta)
    interpretation_log_path: Optional[str] = None
    
    @classmethod
    def create(cls, filename: str, file_size: int, file_type: str, content: bytes) -> 'NormalizedDocument':
        """Create a new normalized document with auto-generated ID and metadata."""
        # Generate document ID using UUID short hash
        doc_id = str(uuid.uuid4())[:8]
        
        # Generate content hash
        content_hash = hashlib.sha256(content).hexdigest()
        
        # Create ingest metadata
        ingest_metadata = IngestMetadata(
            filename=filename,
            file_size=file_size,
            file_type=file_type,
            uploaded_at=datetime.now(),
            content_hash=content_hash
        )
        
        return cls(
            doc_id=doc_id,
            ingest_metadata=ingest_metadata,
            processing_meta=ProcessingMeta()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        def _convert_value(obj):
            if hasattr(obj, '__dict__'):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return {k: _convert_value(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [_convert_value(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _convert_value(v) for k, v in obj.items()}
            else:
                return obj
        
        return _convert_value(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NormalizedDocument':
        """Create from dictionary (for JSON deserialization)."""
        # This is a simplified version - in production, you'd want more robust deserialization
        # For now, we'll implement basic reconstruction
        doc = cls(
            doc_id=data['doc_id'],
            ingest_metadata=IngestMetadata(**data['ingest_metadata']),
            processing_meta=ProcessingMeta(**data.get('processing_meta', {}))
        )
        
        # Add sections, key_values, tables, chunks if present
        if 'sections' in data:
            doc.sections = [Section(**section) for section in data['sections']]
        if 'key_values' in data:
            doc.key_values = [KeyValue(**kv) for kv in data['key_values']]
        if 'tables' in data:
            doc.tables = [Table(**table) for table in data['tables']]
        if 'chunks' in data:
            doc.chunks = [Chunk(**chunk) for chunk in data['chunks']]
            
        doc.interpretation_log_path = data.get('interpretation_log_path')
        
        return doc


def calculate_coverage_stats(doc: NormalizedDocument) -> Dict[str, Any]:
    """Calculate coverage statistics for a normalized document."""
    total_fields = len(doc.key_values)
    rule_extracted = len([kv for kv in doc.key_values if kv.extraction_method == "rule"])
    model_extracted = len([kv for kv in doc.key_values if kv.extraction_method == "model"])
    
    return {
        "required_fields_total": total_fields,
        "extracted": total_fields,
        "rule_based": rule_extracted,
        "model_based": model_extracted,
        "rule_coverage": rule_extracted / total_fields if total_fields > 0 else 0.0,
        "model_coverage": model_extracted / total_fields if total_fields > 0 else 0.0
    }