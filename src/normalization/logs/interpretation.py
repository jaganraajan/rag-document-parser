"""
Interpretation Logging System.

This module provides JSONL event logging for the document processing pipeline,
capturing each pipeline decision with stage, action, detail, decision, and cost information.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class LogEvent:
    """Represents a single pipeline log event."""
    timestamp: str
    doc_id: str
    stage: str  # "INGEST", "SIGNATURE", "RULES", "LLM", "FINALIZE"
    action: str
    detail: str
    decision: str
    cost_usd: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class InterpretationLogger:
    """JSONL event logger for pipeline interpretability."""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = logs_dir
        self.current_doc_id: Optional[str] = None
        self.current_log_file: Optional[str] = None
        self._ensure_logs_dir()
    
    def _ensure_logs_dir(self):
        """Ensure logs directory exists."""
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def start_document_logging(self, doc_id: str) -> str:
        """Start logging for a new document."""
        self.current_doc_id = doc_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = os.path.join(self.logs_dir, f"{doc_id}_{timestamp}.jsonl")
        
        # Log document start
        self.log_event(
            stage="INGEST",
            action="start_processing",
            detail=f"Started processing document {doc_id}",
            decision="proceed"
        )
        
        return self.current_log_file
    
    def log_event(self, stage: str, action: str, detail: str, decision: str, 
                  cost_usd: float = 0.0, metadata: Dict[str, Any] = None):
        """Log a pipeline event."""
        if not self.current_doc_id:
            logger.warning("No active document for logging")
            return
        
        event = LogEvent(
            timestamp=datetime.now().isoformat(),
            doc_id=self.current_doc_id,
            stage=stage,
            action=action,
            detail=detail,
            decision=decision,
            cost_usd=cost_usd,
            metadata=metadata or {}
        )
        
        self._write_event(event)
    
    def _write_event(self, event: LogEvent):
        """Write event to JSONL file."""
        if not self.current_log_file:
            logger.warning("No active log file")
            return
        
        try:
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                json.dump(asdict(event), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logger.error(f"Error writing log event: {e}")
    
    def log_ingest(self, filename: str, file_size: int, file_type: str):
        """Log document ingestion."""
        self.log_event(
            stage="INGEST",
            action="file_loaded",
            detail=f"Loaded file: {filename}",
            decision="proceed",
            metadata={
                "filename": filename,
                "file_size": file_size,
                "file_type": file_type
            }
        )
    
    def log_signature_creation(self, signature_id: str, is_new: bool, similarity_score: float):
        """Log signature creation or matching."""
        action = "create_signature" if is_new else "match_signature"
        detail = f"{'Created new' if is_new else 'Matched existing'} signature: {signature_id}"
        
        self.log_event(
            stage="SIGNATURE",
            action=action,
            detail=detail,
            decision="proceed",
            metadata={
                "signature_id": signature_id,
                "is_new": is_new,
                "similarity_score": similarity_score
            }
        )
    
    def log_rules_application(self, rules_applied: List[str], fields_extracted: int):
        """Log rules engine application."""
        self.log_event(
            stage="RULES",
            action="apply_rules",
            detail=f"Applied {len(rules_applied)} rule sets, extracted {fields_extracted} fields",
            decision="proceed",
            metadata={
                "rules_applied": rules_applied,
                "fields_extracted": fields_extracted
            }
        )
    
    def log_field_extraction(self, field_name: str, value: Any, method: str, confidence: float):
        """Log individual field extraction."""
        self.log_event(
            stage="RULES" if method == "rule" else "LLM",
            action="extract_field",
            detail=f"Extracted {field_name}: {value}",
            decision="field_extracted",
            metadata={
                "field_name": field_name,
                "value": str(value),
                "method": method,
                "confidence": confidence
            }
        )
    
    def log_llm_invocation(self, field_name: str, success: bool, cost_usd: float, model_name: str):
        """Log LLM model invocation."""
        decision = "field_extracted" if success else "extraction_failed"
        detail = f"LLM extraction for {field_name}: {'success' if success else 'failed'}"
        
        self.log_event(
            stage="LLM",
            action="model_call",
            detail=detail,
            decision=decision,
            cost_usd=cost_usd,
            metadata={
                "field_name": field_name,
                "model_name": model_name,
                "success": success
            }
        )
    
    def log_missing_fields(self, missing_fields: List[str], require_llm: bool):
        """Log missing fields detection."""
        decision = "invoke_llm" if require_llm else "accept_partial"
        detail = f"Found {len(missing_fields)} missing fields: {missing_fields}"
        
        self.log_event(
            stage="RULES",
            action="check_coverage",
            detail=detail,
            decision=decision,
            metadata={
                "missing_fields": missing_fields,
                "require_llm": require_llm
            }
        )
    
    def log_cost_limit(self, estimated_cost: float, limit: float, fields_skipped: List[str]):
        """Log cost limit enforcement."""
        self.log_event(
            stage="LLM",
            action="enforce_cost_limit",
            detail=f"Estimated cost ${estimated_cost:.4f} exceeds limit ${limit:.4f}",
            decision="skip_fields",
            metadata={
                "estimated_cost": estimated_cost,
                "cost_limit": limit,
                "fields_skipped": fields_skipped
            }
        )
    
    def log_processing_complete(self, total_cost: float, processing_time: float, 
                              fields_extracted: int, coverage_stats: Dict[str, Any]):
        """Log processing completion."""
        self.log_event(
            stage="FINALIZE",
            action="complete_processing",
            detail=f"Processing complete: {fields_extracted} fields extracted",
            decision="success",
            cost_usd=total_cost,
            metadata={
                "processing_time_seconds": processing_time,
                "total_fields": fields_extracted,
                "coverage_stats": coverage_stats
            }
        )
    
    def log_error(self, stage: str, error_type: str, error_message: str, metadata: Dict[str, Any] = None):
        """Log an error during processing."""
        self.log_event(
            stage=stage,
            action="error",
            detail=f"{error_type}: {error_message}",
            decision="handle_error",
            metadata={
                "error_type": error_type,
                "error_message": error_message,
                **(metadata or {})
            }
        )
    
    def finish_document_logging(self):
        """Finish logging for current document."""
        if self.current_doc_id:
            self.log_event(
                stage="FINALIZE",
                action="end_processing",
                detail=f"Finished processing document {self.current_doc_id}",
                decision="complete"
            )
        
        self.current_doc_id = None
        self.current_log_file = None
    
    def read_log_file(self, log_file_path: str) -> List[LogEvent]:
        """Read and parse a JSONL log file."""
        events = []
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        event = LogEvent(**data)
                        events.append(event)
        except Exception as e:
            logger.error(f"Error reading log file {log_file_path}: {e}")
        
        return events
    
    def get_document_logs(self, doc_id: str) -> List[LogEvent]:
        """Get all log events for a specific document."""
        events = []
        
        # Find log files for this document
        for filename in os.listdir(self.logs_dir):
            if filename.startswith(doc_id) and filename.endswith('.jsonl'):
                filepath = os.path.join(self.logs_dir, filename)
                file_events = self.read_log_file(filepath)
                events.extend(file_events)
        
        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        return events
    
    def get_recent_logs(self, limit: int = 100) -> List[LogEvent]:
        """Get recent log events across all documents."""
        all_events = []
        
        # Read all log files
        for filename in os.listdir(self.logs_dir):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(self.logs_dir, filename)
                file_events = self.read_log_file(filepath)
                all_events.extend(file_events)
        
        # Sort by timestamp and return recent
        all_events.sort(key=lambda e: e.timestamp, reverse=True)
        return all_events[:limit]
    
    def get_cost_summary(self, doc_id: str = None) -> Dict[str, Any]:
        """Get cost summary for a document or all documents."""
        if doc_id:
            events = self.get_document_logs(doc_id)
        else:
            events = self.get_recent_logs(limit=10000)  # Large limit to get all
        
        total_cost = sum(event.cost_usd for event in events)
        llm_calls = [event for event in events if event.stage == "LLM" and event.action == "model_call"]
        
        return {
            "total_cost_usd": total_cost,
            "llm_calls": len(llm_calls),
            "successful_calls": len([call for call in llm_calls if call.decision == "field_extracted"]),
            "avg_cost_per_call": total_cost / len(llm_calls) if llm_calls else 0.0,
            "cost_by_stage": self._group_costs_by_stage(events)
        }
    
    def _group_costs_by_stage(self, events: List[LogEvent]) -> Dict[str, float]:
        """Group costs by pipeline stage."""
        costs = {}
        for event in events:
            stage = event.stage
            if stage not in costs:
                costs[stage] = 0.0
            costs[stage] += event.cost_usd
        return costs
    
    def export_logs_csv(self, output_path: str, doc_id: str = None):
        """Export logs to CSV format."""
        import csv
        
        if doc_id:
            events = self.get_document_logs(doc_id)
        else:
            events = self.get_recent_logs(limit=10000)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'doc_id', 'stage', 'action', 'detail', 'decision', 'cost_usd']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for event in events:
                writer.writerow({
                    'timestamp': event.timestamp,
                    'doc_id': event.doc_id,
                    'stage': event.stage,
                    'action': event.action,
                    'detail': event.detail,
                    'decision': event.decision,
                    'cost_usd': event.cost_usd
                })
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days."""
        import time
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        for filename in os.listdir(self.logs_dir):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(self.logs_dir, filename)
                if os.path.getmtime(filepath) < cutoff_time:
                    try:
                        os.remove(filepath)
                        logger.info(f"Removed old log file: {filename}")
                    except Exception as e:
                        logger.error(f"Error removing log file {filename}: {e}")