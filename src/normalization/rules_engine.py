"""
Rules Engine for Hybrid Document Normalization.

This module handles loading global and signature-specific rule overrides,
and applies basic regex/heuristic extraction patterns.
"""

import os
import re
import yaml
from typing import Dict, List, Any, Optional, Pattern
from dataclasses import dataclass
from src.normalization.schema import KeyValue, BoundingBox


@dataclass
class ExtractionRule:
    """Represents a single extraction rule."""
    field_name: str
    pattern: str
    confidence: float = 1.0
    description: str = ""
    extraction_type: str = "regex"  # "regex", "heuristic", "position"
    required: bool = False
    
    def __post_init__(self):
        self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)
    
    def extract(self, text: str) -> List[KeyValue]:
        """Extract values using this rule."""
        matches = []
        
        if self.extraction_type == "regex":
            for match in self.compiled_pattern.finditer(text):
                value = match.group(1) if match.groups() else match.group(0)
                # Clean up the value
                value = value.strip()
                
                # Try to convert to appropriate type
                converted_value = self._convert_value(value)
                
                kv = KeyValue(
                    key=self.field_name,
                    value=converted_value,
                    confidence=self.confidence,
                    extraction_method="rule"
                )
                matches.append(kv)
        
        return matches
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Try to convert to boolean
        if value.lower() in ['true', 'yes', '1']:
            return True
        elif value.lower() in ['false', 'no', '0']:
            return False
        
        # Return as string
        return value


@dataclass
class RuleSet:
    """A collection of extraction rules."""
    name: str
    version: str = "1.0"
    description: str = ""
    rules: List[ExtractionRule] = None
    
    def __post_init__(self):
        if self.rules is None:
            self.rules = []
    
    def add_rule(self, rule: ExtractionRule):
        """Add a rule to this set."""
        self.rules.append(rule)
    
    def extract_all(self, text: str) -> List[KeyValue]:
        """Extract all values using all rules in this set."""
        all_matches = []
        for rule in self.rules:
            matches = rule.extract(text)
            all_matches.extend(matches)
        return all_matches


class RulesEngine:
    """Main rules engine for loading and applying extraction rules."""
    
    def __init__(self, rules_dir: str = "rules"):
        self.rules_dir = rules_dir
        self.global_rules: RuleSet = RuleSet("global")
        self.signature_rules: Dict[str, RuleSet] = {}
        self.load_rules()
    
    def load_rules(self):
        """Load global and signature-specific rules."""
        self._load_global_rules()
        self._load_signature_rules()
    
    def _load_global_rules(self):
        """Load global rules from global_rules.yml."""
        global_rules_path = os.path.join(self.rules_dir, "global_rules.yml")
        
        if os.path.exists(global_rules_path):
            try:
                with open(global_rules_path, 'r') as f:
                    rules_data = yaml.safe_load(f)
                    self.global_rules = self._parse_ruleset(rules_data, "global")
            except Exception as e:
                print(f"Error loading global rules: {e}")
        else:
            # Create default global rules if file doesn't exist
            self._create_default_global_rules()
    
    def _load_signature_rules(self):
        """Load signature-specific rule overrides."""
        overrides_dir = os.path.join(self.rules_dir, "signature_overrides")
        
        if not os.path.exists(overrides_dir):
            return
        
        for filename in os.listdir(overrides_dir):
            if filename.endswith('.yml') or filename.endswith('.yaml'):
                signature_id = filename.split('.')[0]
                filepath = os.path.join(overrides_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        rules_data = yaml.safe_load(f)
                        ruleset = self._parse_ruleset(rules_data, signature_id)
                        self.signature_rules[signature_id] = ruleset
                except Exception as e:
                    print(f"Error loading signature rules for {signature_id}: {e}")
    
    def _parse_ruleset(self, data: Dict[str, Any], name: str) -> RuleSet:
        """Parse ruleset from YAML data."""
        ruleset = RuleSet(
            name=name,
            version=data.get('version', '1.0'),
            description=data.get('description', '')
        )
        
        for rule_data in data.get('rules', []):
            rule = ExtractionRule(
                field_name=rule_data['field_name'],
                pattern=rule_data['pattern'],
                confidence=rule_data.get('confidence', 1.0),
                description=rule_data.get('description', ''),
                extraction_type=rule_data.get('extraction_type', 'regex'),
                required=rule_data.get('required', False)
            )
            ruleset.add_rule(rule)
        
        return ruleset
    
    def _create_default_global_rules(self):
        """Create default global rules for common invoice-like fields."""
        default_rules = [
            ExtractionRule(
                field_name="invoice_number",
                pattern=r"(?:invoice|inv|bill)[\s#:]*([A-Z0-9-]+)",
                confidence=0.9,
                description="Extract invoice number",
                required=True
            ),
            ExtractionRule(
                field_name="total_amount",
                pattern=r"(?:total|amount|sum|due)[\s:]*\$?(\d+\.?\d*)",
                confidence=0.8,
                description="Extract total amount",
                required=True
            ),
            ExtractionRule(
                field_name="date",
                pattern=r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                confidence=0.7,
                description="Extract date in MM/DD/YYYY or similar format"
            ),
            ExtractionRule(
                field_name="vendor_name",
                pattern=r"(?:from|vendor|company)[\s:]*([A-Za-z\s]+)(?:\n|$)",
                confidence=0.6,
                description="Extract vendor name"
            ),
            ExtractionRule(
                field_name="phone_number",
                pattern=r"(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})",
                confidence=0.8,
                description="Extract phone number"
            ),
            ExtractionRule(
                field_name="email",
                pattern=r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                confidence=0.9,
                description="Extract email address"
            )
        ]
        
        for rule in default_rules:
            self.global_rules.add_rule(rule)
        
        # Save default rules to file
        self._save_global_rules()
    
    def _save_global_rules(self):
        """Save global rules to file."""
        os.makedirs(self.rules_dir, exist_ok=True)
        global_rules_path = os.path.join(self.rules_dir, "global_rules.yml")
        
        rules_data = {
            "version": self.global_rules.version,
            "description": self.global_rules.description or "Global extraction rules",
            "rules": [
                {
                    "field_name": rule.field_name,
                    "pattern": rule.pattern,
                    "confidence": rule.confidence,
                    "description": rule.description,
                    "extraction_type": rule.extraction_type,
                    "required": rule.required
                }
                for rule in self.global_rules.rules
            ]
        }
        
        with open(global_rules_path, 'w') as f:
            yaml.dump(rules_data, f, default_flow_style=False, indent=2)
    
    def apply_rules(self, text: str, signature_id: Optional[str] = None) -> List[KeyValue]:
        """Apply rules to extract key-value pairs from text."""
        extracted_values = []
        applied_rules = []
        
        # Apply global rules first
        global_matches = self.global_rules.extract_all(text)
        extracted_values.extend(global_matches)
        applied_rules.append("global")
        
        # Apply signature-specific overrides if available
        if signature_id and signature_id in self.signature_rules:
            signature_matches = self.signature_rules[signature_id].extract_all(text)
            extracted_values.extend(signature_matches)
            applied_rules.append(f"signature_{signature_id}")
        
        # Remove duplicates (keep highest confidence)
        deduplicated = self._deduplicate_values(extracted_values)
        
        return deduplicated, applied_rules
    
    def _deduplicate_values(self, values: List[KeyValue]) -> List[KeyValue]:
        """Remove duplicate values, keeping the one with highest confidence."""
        value_map = {}
        
        for value in values:
            key = value.key
            if key not in value_map or value.confidence > value_map[key].confidence:
                value_map[key] = value
        
        return list(value_map.values())
    
    def get_required_fields(self, signature_id: Optional[str] = None) -> List[str]:
        """Get list of required fields for a signature."""
        required_fields = []
        
        # Global required fields
        for rule in self.global_rules.rules:
            if rule.required:
                required_fields.append(rule.field_name)
        
        # Signature-specific required fields
        if signature_id and signature_id in self.signature_rules:
            for rule in self.signature_rules[signature_id].rules:
                if rule.required:
                    required_fields.append(rule.field_name)
        
        return list(set(required_fields))
    
    def add_signature_rule(self, signature_id: str, rule: ExtractionRule):
        """Add a rule for a specific signature."""
        if signature_id not in self.signature_rules:
            self.signature_rules[signature_id] = RuleSet(f"signature_{signature_id}")
        
        self.signature_rules[signature_id].add_rule(rule)
        
        # Save to file
        self._save_signature_rules(signature_id)
    
    def _save_signature_rules(self, signature_id: str):
        """Save signature-specific rules to file."""
        if signature_id not in self.signature_rules:
            return
        
        overrides_dir = os.path.join(self.rules_dir, "signature_overrides")
        os.makedirs(overrides_dir, exist_ok=True)
        
        filepath = os.path.join(overrides_dir, f"{signature_id}.yml")
        ruleset = self.signature_rules[signature_id]
        
        rules_data = {
            "version": ruleset.version,
            "description": ruleset.description,
            "rules": [
                {
                    "field_name": rule.field_name,
                    "pattern": rule.pattern,
                    "confidence": rule.confidence,
                    "description": rule.description,
                    "extraction_type": rule.extraction_type,
                    "required": rule.required
                }
                for rule in ruleset.rules
            ]
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(rules_data, f, default_flow_style=False, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded rules."""
        return {
            "global_rules_count": len(self.global_rules.rules),
            "signature_overrides_count": len(self.signature_rules),
            "total_rules": len(self.global_rules.rules) + sum(
                len(ruleset.rules) for ruleset in self.signature_rules.values()
            ),
            "signature_ids": list(self.signature_rules.keys())
        }