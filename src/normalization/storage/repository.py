"""
Storage Repository for Normalized Documents.

This module handles persistence of normalized JSON documents with versioning,
metadata indexing, and efficient retrieval operations.
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

from src.normalization.schema import NormalizedDocument, calculate_coverage_stats

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for storing and retrieving normalized documents."""
    
    def __init__(self, storage_dir: str = "outputs"):
        self.storage_dir = storage_dir
        self.index_file = os.path.join(storage_dir, "document_index.json")
        self.index: Dict[str, Dict[str, Any]] = {}
        self._ensure_storage_dir()
        self._load_index()
    
    def _ensure_storage_dir(self):
        """Ensure storage directory exists."""
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def _load_index(self):
        """Load document index from disk."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            except Exception as e:
                logger.error(f"Error loading document index: {e}")
                self.index = {}
        else:
            self.index = {}
    
    def _save_index(self):
        """Save document index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving document index: {e}")
    
    def save_document(self, document: NormalizedDocument) -> str:
        """
        Save a normalized document.
        
        Returns:
            File path where document was saved
        """
        doc_id = document.doc_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{doc_id}_{timestamp}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        try:
            # Save document
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(document.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            
            # Update index
            self._update_index(document, filepath)
            self._save_index()
            
            logger.info(f"Saved document {doc_id} to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving document {doc_id}: {e}")
            raise
    
    def _update_index(self, document: NormalizedDocument, filepath: str):
        """Update document index with new document."""
        doc_id = document.doc_id
        
        # Calculate coverage stats
        coverage_stats = calculate_coverage_stats(document)
        
        index_entry = {
            "doc_id": doc_id,
            "filename": document.ingest_metadata.filename,
            "file_type": document.ingest_metadata.file_type,
            "file_size": document.ingest_metadata.file_size,
            "uploaded_at": document.ingest_metadata.uploaded_at.isoformat(),
            "content_hash": document.ingest_metadata.content_hash,
            "signature_id": document.processing_meta.signature_id,
            "signature_match_score": document.processing_meta.signature_match_score,
            "total_cost_usd": document.processing_meta.total_cost_usd,
            "model_calls_made": document.processing_meta.model_calls_made,
            "coverage_stats": coverage_stats,
            "saved_at": datetime.now().isoformat(),
            "filepath": filepath,
            "sections_count": len(document.sections),
            "key_values_count": len(document.key_values),
            "tables_count": len(document.tables),
            "chunks_count": len(document.chunks)
        }
        
        self.index[doc_id] = index_entry
    
    def get_document(self, doc_id: str) -> Optional[NormalizedDocument]:
        """Retrieve a document by ID."""
        if doc_id not in self.index:
            logger.warning(f"Document {doc_id} not found in index")
            return None
        
        filepath = self.index[doc_id]["filepath"]
        
        if not os.path.exists(filepath):
            logger.error(f"Document file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return NormalizedDocument.from_dict(data)
            
        except Exception as e:
            logger.error(f"Error loading document {doc_id}: {e}")
            return None
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List documents with pagination."""
        # Sort by upload time (most recent first)
        sorted_docs = sorted(
            self.index.values(),
            key=lambda x: x["uploaded_at"],
            reverse=True
        )
        
        return sorted_docs[offset:offset + limit]
    
    def search_documents(self, **filters) -> List[Dict[str, Any]]:
        """
        Search documents by various criteria.
        
        Supported filters:
        - file_type: str
        - signature_id: str
        - min_coverage: float (minimum rule coverage)
        - max_cost: float (maximum processing cost)
        - has_tables: bool
        - date_from: str (ISO format)
        - date_to: str (ISO format)
        """
        results = []
        
        for doc_info in self.index.values():
            if self._matches_filters(doc_info, filters):
                results.append(doc_info)
        
        # Sort by relevance (most recent first)
        results.sort(key=lambda x: x["uploaded_at"], reverse=True)
        return results
    
    def _matches_filters(self, doc_info: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document matches search filters."""
        # File type filter
        if "file_type" in filters:
            if doc_info["file_type"] != filters["file_type"]:
                return False
        
        # Signature filter
        if "signature_id" in filters:
            if doc_info["signature_id"] != filters["signature_id"]:
                return False
        
        # Coverage filter
        if "min_coverage" in filters:
            coverage = doc_info.get("coverage_stats", {}).get("rule_coverage", 0.0)
            if coverage < filters["min_coverage"]:
                return False
        
        # Cost filter
        if "max_cost" in filters:
            if doc_info["total_cost_usd"] > filters["max_cost"]:
                return False
        
        # Tables filter
        if "has_tables" in filters:
            has_tables = doc_info["tables_count"] > 0
            if has_tables != filters["has_tables"]:
                return False
        
        # Date range filters
        doc_date = doc_info["uploaded_at"]
        
        if "date_from" in filters:
            if doc_date < filters["date_from"]:
                return False
        
        if "date_to" in filters:
            if doc_date > filters["date_to"]:
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        if not self.index:
            return {
                "total_documents": 0,
                "total_size_bytes": 0,
                "avg_processing_cost": 0.0,
                "file_types": {},
                "signatures": {}
            }
        
        total_size = sum(doc["file_size"] for doc in self.index.values())
        total_cost = sum(doc["total_cost_usd"] for doc in self.index.values())
        
        # File type distribution
        file_types = {}
        for doc in self.index.values():
            ft = doc["file_type"]
            file_types[ft] = file_types.get(ft, 0) + 1
        
        # Signature distribution
        signatures = {}
        for doc in self.index.values():
            sig = doc.get("signature_id", "unknown")
            signatures[sig] = signatures.get(sig, 0) + 1
        
        return {
            "total_documents": len(self.index),
            "total_size_bytes": total_size,
            "avg_file_size_bytes": total_size / len(self.index),
            "total_processing_cost": total_cost,
            "avg_processing_cost": total_cost / len(self.index),
            "file_types": file_types,
            "signatures": signatures,
            "avg_key_values": sum(doc["key_values_count"] for doc in self.index.values()) / len(self.index),
            "avg_sections": sum(doc["sections_count"] for doc in self.index.values()) / len(self.index)
        }
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its file."""
        if doc_id not in self.index:
            logger.warning(f"Document {doc_id} not found")
            return False
        
        filepath = self.index[doc_id]["filepath"]
        
        try:
            # Remove file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Remove from index
            del self.index[doc_id]
            self._save_index()
            
            logger.info(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def export_document(self, doc_id: str, output_path: str) -> bool:
        """Export a document to a specific path."""
        if doc_id not in self.index:
            return False
        
        filepath = self.index[doc_id]["filepath"]
        
        try:
            shutil.copy2(filepath, output_path)
            return True
        except Exception as e:
            logger.error(f"Error exporting document {doc_id}: {e}")
            return False
    
    def backup_repository(self, backup_dir: str) -> bool:
        """Create a backup of the entire repository."""
        try:
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            
            shutil.copytree(self.storage_dir, backup_dir)
            logger.info(f"Repository backed up to {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def restore_repository(self, backup_dir: str) -> bool:
        """Restore repository from backup."""
        if not os.path.exists(backup_dir):
            logger.error(f"Backup directory not found: {backup_dir}")
            return False
        
        try:
            # Backup current repository
            current_backup = f"{self.storage_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(self.storage_dir, current_backup)
            
            # Restore from backup
            shutil.copytree(backup_dir, self.storage_dir)
            
            # Reload index
            self._load_index()
            
            logger.info(f"Repository restored from {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return False
    
    def cleanup_old_documents(self, days_to_keep: int = 90) -> int:
        """Remove documents older than specified days."""
        import time
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        removed_count = 0
        
        docs_to_remove = []
        
        for doc_id, doc_info in self.index.items():
            doc_time = datetime.fromisoformat(doc_info["uploaded_at"]).timestamp()
            if doc_time < cutoff_time:
                docs_to_remove.append(doc_id)
        
        for doc_id in docs_to_remove:
            if self.delete_document(doc_id):
                removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} old documents")
        return removed_count
    
    def get_document_by_hash(self, content_hash: str) -> Optional[str]:
        """Find document ID by content hash (for deduplication)."""
        for doc_id, doc_info in self.index.items():
            if doc_info["content_hash"] == content_hash:
                return doc_id
        return None
    
    def get_documents_by_signature(self, signature_id: str) -> List[Dict[str, Any]]:
        """Get all documents with a specific signature."""
        return [
            doc_info for doc_info in self.index.values()
            if doc_info.get("signature_id") == signature_id
        ]