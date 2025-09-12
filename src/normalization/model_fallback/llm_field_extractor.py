"""
LLM Field Extractor - Selective AI Fallback Stub.

This module provides a placeholder for AI-powered field extraction that only
triggers when fields are missing after rule-based extraction. It includes
a pluggable architecture for future model provider integration.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.normalization.schema import KeyValue

logger = logging.getLogger(__name__)


@dataclass
class ModelCall:
    """Represents a model API call for cost tracking."""
    timestamp: float
    model_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    field_name: str
    success: bool
    latency_ms: float


class ModelProvider(ABC):
    """Abstract base class for model providers."""
    
    @abstractmethod
    def extract_field(self, field_name: str, context: str, examples: List[str] = None) -> Optional[KeyValue]:
        """Extract a specific field value from context."""
        pass
    
    @abstractmethod
    def estimate_cost(self, input_text: str, field_name: str) -> float:
        """Estimate the cost of extracting a field."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get provider name."""
        pass


class DummyModelProvider(ModelProvider):
    """Dummy model provider for testing and placeholder functionality."""
    
    def __init__(self, delay_seconds: float = 0.5):
        self.delay_seconds = delay_seconds
        self.call_count = 0
    
    def extract_field(self, field_name: str, context: str, examples: List[str] = None) -> Optional[KeyValue]:
        """Extract field using dummy logic."""
        start_time = time.time()
        self.call_count += 1
        
        # Simulate API delay
        time.sleep(self.delay_seconds)
        
        # Generate dummy value based on field name
        dummy_value = self._generate_dummy_value(field_name, context)
        
        if dummy_value:
            return KeyValue(
                key=field_name,
                value=dummy_value,
                confidence=0.5,  # Low confidence for dummy data
                extraction_method="model"
            )
        
        return None
    
    def _generate_dummy_value(self, field_name: str, context: str) -> Optional[Union[str, float, int]]:
        """Generate dummy values based on field name patterns."""
        field_lower = field_name.lower()
        
        # Dummy values based on common field patterns
        if 'invoice' in field_lower and 'number' in field_lower:
            return f"INV-{self.call_count:04d}"
        elif 'total' in field_lower or 'amount' in field_lower:
            return round(100.0 + (self.call_count * 23.45), 2)
        elif 'date' in field_lower:
            return "2024-01-15"
        elif 'vendor' in field_lower or 'company' in field_lower:
            return f"Dummy Company {self.call_count}"
        elif 'phone' in field_lower:
            return f"555-{self.call_count:04d}"
        elif 'email' in field_lower:
            return f"dummy{self.call_count}@example.com"
        elif 'address' in field_lower:
            return f"{self.call_count} Main St, Anytown, ST 12345"
        else:
            # Generic dummy value
            return f"DUMMY_{field_name.upper()}_{self.call_count}"
    
    def estimate_cost(self, input_text: str, field_name: str) -> float:
        """Estimate cost - dummy implementation."""
        # Rough token estimation: ~4 chars per token
        input_tokens = len(input_text) // 4
        # Assume output is much smaller
        output_tokens = 10
        
        # Dummy pricing: $0.001 per 1000 input tokens, $0.002 per 1000 output tokens
        cost = (input_tokens / 1000 * 0.001) + (output_tokens / 1000 * 0.002)
        return round(cost, 6)
    
    def get_name(self) -> str:
        return "dummy_provider"


class OpenAIModelProvider(ModelProvider):
    """OpenAI model provider stub (requires openai package)."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if OpenAI is available."""
        try:
            import openai
            return bool(self.api_key)
        except ImportError:
            logger.warning("OpenAI package not available")
            return False
    
    def extract_field(self, field_name: str, context: str, examples: List[str] = None) -> Optional[KeyValue]:
        """Extract field using OpenAI API."""
        if not self.available:
            logger.warning("OpenAI not available, using dummy extraction")
            return DummyModelProvider().extract_field(field_name, context, examples)
        
        try:
            import openai
            # Implementation would go here
            # For now, return dummy value
            return DummyModelProvider().extract_field(field_name, context, examples)
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            return None
    
    def estimate_cost(self, input_text: str, field_name: str) -> float:
        """Estimate OpenAI API cost."""
        # Real implementation would use actual pricing
        return DummyModelProvider().estimate_cost(input_text, field_name)
    
    def get_name(self) -> str:
        return f"openai_{self.model}"


class GeminiModelProvider(ModelProvider):
    """Google Gemini model provider stub."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-pro"):
        self.api_key = api_key
        self.model = model
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Gemini is available."""
        try:
            import google.generativeai as genai
            return bool(self.api_key)
        except ImportError:
            logger.warning("Google Generative AI package not available")
            return False
    
    def extract_field(self, field_name: str, context: str, examples: List[str] = None) -> Optional[KeyValue]:
        """Extract field using Gemini API."""
        if not self.available:
            logger.warning("Gemini not available, using dummy extraction")
            return DummyModelProvider().extract_field(field_name, context, examples)
        
        try:
            # Implementation would go here
            # For now, return dummy value
            return DummyModelProvider().extract_field(field_name, context, examples)
        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            return None
    
    def estimate_cost(self, input_text: str, field_name: str) -> float:
        """Estimate Gemini API cost."""
        # Real implementation would use actual pricing
        return DummyModelProvider().estimate_cost(input_text, field_name)
    
    def get_name(self) -> str:
        return f"gemini_{self.model}"


class LLMFieldExtractor:
    """Main LLM field extractor with selective AI fallback."""
    
    def __init__(self, provider: ModelProvider = None, cache_enabled: bool = True):
        self.provider = provider or DummyModelProvider()
        self.cache_enabled = cache_enabled
        self.cache: Dict[str, KeyValue] = {}
        self.call_history: List[ModelCall] = []
        self.total_cost_usd = 0.0
    
    def extract_missing_fields(self, required_fields: List[str], extracted_fields: List[KeyValue], 
                             context: str, max_cost_usd: float = 1.0) -> List[KeyValue]:
        """
        Extract missing fields using LLM fallback.
        
        Args:
            required_fields: List of field names that should be extracted
            extracted_fields: Fields already extracted by rules
            context: Full text context for extraction
            max_cost_usd: Maximum cost limit for LLM calls
        
        Returns:
            List of newly extracted KeyValue objects
        """
        # Find missing fields
        extracted_field_names = {kv.key for kv in extracted_fields}
        missing_fields = [field for field in required_fields if field not in extracted_field_names]
        
        if not missing_fields:
            logger.info("No missing fields to extract")
            return []
        
        logger.info(f"Extracting {len(missing_fields)} missing fields: {missing_fields}")
        
        # Estimate total cost
        estimated_cost = sum(
            self.provider.estimate_cost(context, field) for field in missing_fields
        )
        
        if estimated_cost > max_cost_usd:
            logger.warning(f"Estimated cost ${estimated_cost:.4f} exceeds limit ${max_cost_usd:.4f}")
            # Extract only the most important fields within budget
            missing_fields = self._prioritize_fields(missing_fields, context, max_cost_usd)
        
        # Extract missing fields
        newly_extracted = []
        
        for field_name in missing_fields:
            if self.total_cost_usd >= max_cost_usd:
                logger.warning(f"Cost limit reached, skipping field: {field_name}")
                break
            
            # Check cache first
            cache_key = self._get_cache_key(field_name, context)
            if self.cache_enabled and cache_key in self.cache:
                logger.info(f"Using cached result for field: {field_name}")
                newly_extracted.append(self.cache[cache_key])
                continue
            
            # Extract using model
            start_time = time.time()
            estimated_cost = self.provider.estimate_cost(context, field_name)
            
            try:
                result = self.provider.extract_field(field_name, context)
                
                if result:
                    newly_extracted.append(result)
                    
                    # Cache result
                    if self.cache_enabled:
                        self.cache[cache_key] = result
                
                # Record call
                call = ModelCall(
                    timestamp=start_time,
                    model_name=self.provider.get_name(),
                    input_tokens=len(context) // 4,  # Rough estimation
                    output_tokens=len(str(result.value)) // 4 if result else 0,
                    cost_usd=estimated_cost,
                    field_name=field_name,
                    success=result is not None,
                    latency_ms=(time.time() - start_time) * 1000
                )
                self.call_history.append(call)
                self.total_cost_usd += estimated_cost
                
                logger.info(f"Extracted field '{field_name}': {result.value if result else 'None'} "
                           f"(cost: ${estimated_cost:.4f})")
                
            except Exception as e:
                logger.error(f"Error extracting field '{field_name}': {e}")
                
                # Record failed call
                call = ModelCall(
                    timestamp=start_time,
                    model_name=self.provider.get_name(),
                    input_tokens=len(context) // 4,
                    output_tokens=0,
                    cost_usd=0.0,
                    field_name=field_name,
                    success=False,
                    latency_ms=(time.time() - start_time) * 1000
                )
                self.call_history.append(call)
        
        logger.info(f"Extracted {len(newly_extracted)} fields using LLM "
                   f"(total cost: ${self.total_cost_usd:.4f})")
        
        return newly_extracted
    
    def _get_cache_key(self, field_name: str, context: str) -> str:
        """Generate cache key for field extraction."""
        import hashlib
        content_hash = hashlib.md5(context.encode()).hexdigest()[:16]
        return f"{field_name}:{content_hash}"
    
    def _prioritize_fields(self, fields: List[str], context: str, max_cost: float) -> List[str]:
        """Prioritize fields to extract within cost limit."""
        # Simple prioritization: sort by estimated cost and take what fits
        field_costs = [(field, self.provider.estimate_cost(context, field)) for field in fields]
        field_costs.sort(key=lambda x: x[1])  # Sort by cost
        
        prioritized = []
        total_cost = 0.0
        
        for field, cost in field_costs:
            if total_cost + cost <= max_cost:
                prioritized.append(field)
                total_cost += cost
            else:
                break
        
        return prioritized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        if not self.call_history:
            return {
                "total_calls": 0,
                "total_cost_usd": 0.0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0
            }
        
        successful_calls = [call for call in self.call_history if call.success]
        
        return {
            "total_calls": len(self.call_history),
            "successful_calls": len(successful_calls),
            "total_cost_usd": self.total_cost_usd,
            "success_rate": len(successful_calls) / len(self.call_history),
            "avg_latency_ms": sum(call.latency_ms for call in self.call_history) / len(self.call_history),
            "total_input_tokens": sum(call.input_tokens for call in self.call_history),
            "total_output_tokens": sum(call.output_tokens for call in self.call_history),
            "cache_size": len(self.cache) if self.cache_enabled else 0
        }
    
    def clear_cache(self):
        """Clear the extraction cache."""
        self.cache.clear()
    
    def reset_stats(self):
        """Reset call history and cost tracking."""
        self.call_history.clear()
        self.total_cost_usd = 0.0


def create_provider(provider_type: str = "dummy", **kwargs) -> ModelProvider:
    """Factory function to create model providers."""
    if provider_type == "dummy":
        return DummyModelProvider(**kwargs)
    elif provider_type == "openai":
        return OpenAIModelProvider(**kwargs)
    elif provider_type == "gemini":
        return GeminiModelProvider(**kwargs)
    else:
        logger.warning(f"Unknown provider type: {provider_type}, using dummy")
        return DummyModelProvider()