"""
Layout Signature Learning and Matching System.

This module handles generating layout signatures from quantized structural tokens
(page, type, bbox buckets, token counts) with Jaccard-based matching, and persists
signature definitions with versioning.
"""

import json
import hashlib
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime


@dataclass
class StructuralToken:
    """Represents a quantized structural element."""
    page: int
    element_type: str  # "text", "image", "table", etc.
    bbox_bucket: Tuple[int, int, int, int]  # quantized to 0-1000 grid
    token_count: int
    content_hash: Optional[str] = None


@dataclass
class LayoutSignature:
    """Represents a layout signature for document matching."""
    signature_id: str
    tokens: List[StructuralToken] = field(default_factory=list)
    hash_value: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    document_count: int = 1
    sample_filenames: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.hash_value and self.tokens:
            self.hash_value = self.compute_hash()
    
    def compute_hash(self) -> str:
        """Compute SHA1 hash of sorted tokens."""
        if not self.tokens:
            return ""
        
        # Sort tokens for consistent hashing
        sorted_tokens = sorted(self.tokens, key=lambda t: (
            t.page, t.element_type, t.bbox_bucket, t.token_count
        ))
        
        # Create hash string
        hash_string = "|".join([
            f"{t.page}:{t.element_type}:{t.bbox_bucket}:{t.token_count}"
            for t in sorted_tokens
        ])
        
        return hashlib.sha1(hash_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "signature_id": self.signature_id,
            "tokens": [
                {
                    "page": t.page,
                    "element_type": t.element_type,
                    "bbox_bucket": t.bbox_bucket,
                    "token_count": t.token_count,
                    "content_hash": t.content_hash
                }
                for t in self.tokens
            ],
            "hash_value": self.hash_value,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "document_count": self.document_count,
            "sample_filenames": self.sample_filenames
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayoutSignature':
        """Create from dictionary."""
        tokens = [
            StructuralToken(
                page=t["page"],
                element_type=t["element_type"],
                bbox_bucket=tuple(t["bbox_bucket"]),
                token_count=t["token_count"],
                content_hash=t.get("content_hash")
            )
            for t in data["tokens"]
        ]
        
        return cls(
            signature_id=data["signature_id"],
            tokens=tokens,
            hash_value=data["hash_value"],
            created_at=datetime.fromisoformat(data["created_at"]),
            version=data["version"],
            document_count=data["document_count"],
            sample_filenames=data["sample_filenames"]
        )


class SignatureManager:
    """Manages layout signatures with persistence and matching."""
    
    def __init__(self, signatures_dir: str = "signatures"):
        self.signatures_dir = signatures_dir
        self.signatures: Dict[str, LayoutSignature] = {}
        self.load_signatures()
    
    def load_signatures(self):
        """Load existing signatures from disk."""
        if not os.path.exists(self.signatures_dir):
            os.makedirs(self.signatures_dir, exist_ok=True)
            return
        
        for filename in os.listdir(self.signatures_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.signatures_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        signature = LayoutSignature.from_dict(data)
                        self.signatures[signature.signature_id] = signature
                except Exception as e:
                    print(f"Error loading signature {filename}: {e}")
    
    def save_signature(self, signature: LayoutSignature):
        """Save signature to disk."""
        os.makedirs(self.signatures_dir, exist_ok=True)
        filename = f"{signature.signature_id}.json"
        filepath = os.path.join(self.signatures_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(signature.to_dict(), f, indent=2)
        
        self.signatures[signature.signature_id] = signature
    
    def quantize_bbox(self, bbox: Tuple[float, float, float, float], 
                     page_width: float, page_height: float) -> Tuple[int, int, int, int]:
        """Quantize bounding box coordinates to 0-1000 grid."""
        x1, y1, x2, y2 = bbox
        
        # Normalize to 0-1 range
        norm_x1 = x1 / page_width if page_width > 0 else 0
        norm_y1 = y1 / page_height if page_height > 0 else 0
        norm_x2 = x2 / page_width if page_width > 0 else 0
        norm_y2 = y2 / page_height if page_height > 0 else 0
        
        # Quantize to 0-1000 grid
        return (
            int(norm_x1 * 1000),
            int(norm_y1 * 1000),
            int(norm_x2 * 1000),
            int(norm_y2 * 1000)
        )
    
    def create_tokens_from_layout(self, layout_elements: List[Dict[str, Any]]) -> List[StructuralToken]:
        """Create structural tokens from layout elements."""
        tokens = []
        
        for element in layout_elements:
            # Extract element information
            page = element.get('page', 1)
            element_type = element.get('type', 'text')
            bbox = element.get('bbox', (0, 0, 100, 100))
            content = element.get('content', '')
            page_width = element.get('page_width', 612)  # Default PDF page width
            page_height = element.get('page_height', 792)  # Default PDF page height
            
            # Quantize bounding box
            bbox_bucket = self.quantize_bbox(bbox, page_width, page_height)
            
            # Estimate token count (rough approximation)
            token_count = len(content.split()) if content else 0
            
            # Create content hash for exact matching if needed
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8] if content else None
            
            token = StructuralToken(
                page=page,
                element_type=element_type,
                bbox_bucket=bbox_bucket,
                token_count=token_count,
                content_hash=content_hash
            )
            tokens.append(token)
        
        return tokens
    
    def compute_jaccard_similarity(self, tokens1: List[StructuralToken], 
                                 tokens2: List[StructuralToken]) -> float:
        """Compute Jaccard similarity between two sets of tokens."""
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        # Convert tokens to comparable tuples
        set1 = set((t.page, t.element_type, t.bbox_bucket, t.token_count) for t in tokens1)
        set2 = set((t.page, t.element_type, t.bbox_bucket, t.token_count) for t in tokens2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def find_matching_signature(self, tokens: List[StructuralToken], 
                              threshold: float = 0.85) -> Optional[LayoutSignature]:
        """Find a matching signature based on Jaccard similarity."""
        best_match = None
        best_score = 0.0
        
        for signature in self.signatures.values():
            similarity = self.compute_jaccard_similarity(tokens, signature.tokens)
            if similarity >= threshold and similarity > best_score:
                best_match = signature
                best_score = similarity
        
        return best_match
    
    def create_or_match_signature(self, layout_elements: List[Dict[str, Any]], 
                                filename: str, threshold: float = 0.85) -> Tuple[LayoutSignature, float]:
        """Create a new signature or match existing one."""
        tokens = self.create_tokens_from_layout(layout_elements)
        
        # Try to find existing signature
        existing = self.find_matching_signature(tokens, threshold)
        
        if existing:
            # Update existing signature
            if filename not in existing.sample_filenames:
                existing.sample_filenames.append(filename)
                existing.document_count += 1
                self.save_signature(existing)
            
            similarity = self.compute_jaccard_similarity(tokens, existing.tokens)
            return existing, similarity
        else:
            # Create new signature
            signature_id = hashlib.sha1(f"{datetime.now().isoformat()}_{filename}".encode()).hexdigest()[:12]
            
            new_signature = LayoutSignature(
                signature_id=signature_id,
                tokens=tokens,
                sample_filenames=[filename]
            )
            
            self.save_signature(new_signature)
            return new_signature, 1.0
    
    def get_signature_stats(self) -> Dict[str, Any]:
        """Get statistics about all signatures."""
        if not self.signatures:
            return {"total_signatures": 0, "total_documents": 0}
        
        total_docs = sum(sig.document_count for sig in self.signatures.values())
        
        return {
            "total_signatures": len(self.signatures),
            "total_documents": total_docs,
            "signatures": [
                {
                    "id": sig.signature_id,
                    "document_count": sig.document_count,
                    "created_at": sig.created_at.isoformat(),
                    "sample_filenames": sig.sample_filenames[:3]  # Show first 3
                }
                for sig in self.signatures.values()
            ]
        }