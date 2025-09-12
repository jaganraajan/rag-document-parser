"""
Email Extractor - RFC822 Email Parsing to Layout Elements.

This module handles parsing email files (.eml, .msg) and converting them
to the normalized document structure with layout elements.
"""

import email
import email.utils
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import re

from src.normalization.schema import Section, BoundingBox, KeyValue

logger = logging.getLogger(__name__)


class EmailExtractor:
    """Email parser that converts RFC822 emails to layout elements."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_from_email(self, file_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract content and layout from email file.
        
        Returns:
            Tuple of (full_text, layout_elements, metadata)
        """
        try:
            with open(file_path, 'rb') as f:
                msg = email.message_from_bytes(f.read())
            
            # Extract email components
            headers = self._extract_headers(msg)
            body = self._extract_body(msg)
            attachments = self._extract_attachments(msg)
            
            # Create layout elements
            layout_elements = self._create_layout_elements(headers, body, attachments)
            
            # Combine all text
            full_text = self._create_full_text(headers, body)
            
            # Create metadata
            metadata = self._create_metadata(msg, headers, attachments)
            
            return full_text, layout_elements, metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting email {file_path}: {e}")
            raise
    
    def _extract_headers(self, msg: email.message.Message) -> Dict[str, str]:
        """Extract email headers."""
        headers = {}
        
        # Standard headers
        standard_headers = [
            'From', 'To', 'Cc', 'Bcc', 'Subject', 'Date', 
            'Message-ID', 'Reply-To', 'Return-Path'
        ]
        
        for header in standard_headers:
            value = msg.get(header)
            if value:
                # Decode header if needed
                decoded = email.utils.parseaddr(value) if header in ['From', 'To', 'Reply-To'] else value
                headers[header.lower()] = str(decoded) if decoded else value
        
        # Additional headers
        for key, value in msg.items():
            if key not in headers:
                headers[key.lower()] = value
        
        return headers
    
    def _extract_body(self, msg: email.message.Message) -> Dict[str, str]:
        """Extract email body content."""
        body = {
            'text': '',
            'html': '',
            'content_type': 'text/plain'
        }
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                
                if content_type == 'text/plain':
                    body['text'] += self._decode_content(part)
                elif content_type == 'text/html':
                    body['html'] += self._decode_content(part)
        else:
            content_type = msg.get_content_type()
            body['content_type'] = content_type
            content = self._decode_content(msg)
            
            if content_type == 'text/html':
                body['html'] = content
                # Strip HTML tags for text version
                body['text'] = self._strip_html(content)
            else:
                body['text'] = content
        
        return body
    
    def _decode_content(self, part: email.message.Message) -> str:
        """Decode email part content."""
        try:
            payload = part.get_payload(decode=True)
            if payload:
                charset = part.get_content_charset() or 'utf-8'
                return payload.decode(charset, errors='ignore')
        except Exception as e:
            self.logger.warning(f"Error decoding email content: {e}")
        
        return ''
    
    def _strip_html(self, html_content: str) -> str:
        """Strip HTML tags to get plain text."""
        import re
        # Simple HTML tag removal - in production, use BeautifulSoup
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', html_content)
        # Replace common HTML entities
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        return text.strip()
    
    def _extract_attachments(self, msg: email.message.Message) -> List[Dict[str, Any]]:
        """Extract attachment information."""
        attachments = []
        
        if msg.is_multipart():
            for part in msg.walk():
                disposition = part.get('Content-Disposition')
                if disposition and disposition.startswith('attachment'):
                    filename = part.get_filename()
                    if filename:
                        attachment = {
                            'filename': filename,
                            'content_type': part.get_content_type(),
                            'size': len(part.get_payload(decode=True) or b'')
                        }
                        attachments.append(attachment)
        
        return attachments
    
    def _create_layout_elements(self, headers: Dict[str, str], body: Dict[str, str], 
                              attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create layout elements from email components."""
        elements = []
        y_position = 0
        page_width = 800.0  # Standard email width
        line_height = 20
        
        # Email headers as layout elements
        header_order = ['from', 'to', 'cc', 'subject', 'date']
        
        for header in header_order:
            if header in headers:
                element = {
                    'content': f"{header.title()}: {headers[header]}",
                    'type': 'header',
                    'bbox': (10, y_position, page_width - 10, y_position + line_height),
                    'page': 1,
                    'page_width': page_width,
                    'page_height': 1000.0,  # Estimated height
                    'header_type': header
                }
                elements.append(element)
                y_position += line_height + 5
        
        # Separator line
        y_position += 10
        elements.append({
            'content': '─' * 80,
            'type': 'separator',
            'bbox': (10, y_position, page_width - 10, y_position + 5),
            'page': 1,
            'page_width': page_width,
            'page_height': 1000.0
        })
        y_position += 15
        
        # Email body
        body_text = body.get('text', '')
        if body_text:
            # Split into paragraphs
            paragraphs = body_text.split('\n\n')
            
            for para in paragraphs:
                if para.strip():
                    # Estimate height based on text length
                    estimated_height = max(line_height, (len(para) // 80 + 1) * line_height)
                    
                    element = {
                        'content': para.strip(),
                        'type': 'text',
                        'bbox': (10, y_position, page_width - 10, y_position + estimated_height),
                        'page': 1,
                        'page_width': page_width,
                        'page_height': 1000.0
                    }
                    elements.append(element)
                    y_position += estimated_height + 10
        
        # Attachments
        if attachments:
            y_position += 20
            elements.append({
                'content': 'Attachments:',
                'type': 'header',
                'bbox': (10, y_position, page_width - 10, y_position + line_height),
                'page': 1,
                'page_width': page_width,
                'page_height': 1000.0,
                'header_type': 'attachments'
            })
            y_position += line_height + 5
            
            for attachment in attachments:
                content = f"📎 {attachment['filename']} ({attachment['content_type']}, {attachment['size']} bytes)"
                element = {
                    'content': content,
                    'type': 'attachment',
                    'bbox': (20, y_position, page_width - 20, y_position + line_height),
                    'page': 1,
                    'page_width': page_width,
                    'page_height': 1000.0,
                    'attachment_info': attachment
                }
                elements.append(element)
                y_position += line_height + 2
        
        # Update page height for all elements
        final_height = max(y_position + 50, 1000.0)
        for element in elements:
            element['page_height'] = final_height
        
        return elements
    
    def _create_full_text(self, headers: Dict[str, str], body: Dict[str, str]) -> str:
        """Create full text representation of email."""
        lines = []
        
        # Add headers
        header_order = ['from', 'to', 'cc', 'subject', 'date']
        for header in header_order:
            if header in headers:
                lines.append(f"{header.title()}: {headers[header]}")
        
        lines.append('')  # Empty line separator
        
        # Add body
        body_text = body.get('text', '')
        if body_text:
            lines.append(body_text)
        
        return '\n'.join(lines)
    
    def _create_metadata(self, msg: email.message.Message, headers: Dict[str, str], 
                        attachments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create metadata for email."""
        metadata = {
            'message_type': 'email',
            'has_attachments': len(attachments) > 0,
            'attachment_count': len(attachments),
            'is_multipart': msg.is_multipart(),
            'content_type': msg.get_content_type()
        }
        
        # Parse date
        if 'date' in headers:
            try:
                parsed_date = email.utils.parsedate_to_datetime(headers['date'])
                metadata['parsed_date'] = parsed_date.isoformat()
            except Exception:
                pass
        
        # Extract email addresses
        for field in ['from', 'to', 'cc']:
            if field in headers:
                addresses = self._extract_email_addresses(headers[field])
                metadata[f'{field}_addresses'] = addresses
        
        return metadata
    
    def _extract_email_addresses(self, header_value: str) -> List[str]:
        """Extract email addresses from header value."""
        addresses = []
        # Simple email extraction - in production, use email.utils.getaddresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, header_value)
        return matches
    
    def convert_to_sections(self, layout_elements: List[Dict[str, Any]]) -> List[Section]:
        """Convert email layout elements to document sections."""
        sections = []
        
        # Group elements by type
        headers = [e for e in layout_elements if e.get('type') == 'header']
        text_elements = [e for e in layout_elements if e.get('type') == 'text']
        attachments = [e for e in layout_elements if e.get('type') == 'attachment']
        
        # Create header section
        if headers:
            header_content = '\n'.join(e['content'] for e in headers)
            bbox = BoundingBox(
                x1=headers[0]['bbox'][0], y1=headers[0]['bbox'][1],
                x2=headers[-1]['bbox'][2], y2=headers[-1]['bbox'][3],
                page=1
            )
            sections.append(Section(
                title="Email Headers",
                content=header_content,
                level=1,
                bbox=bbox
            ))
        
        # Create body section
        if text_elements:
            body_content = '\n\n'.join(e['content'] for e in text_elements)
            bbox = BoundingBox(
                x1=text_elements[0]['bbox'][0], y1=text_elements[0]['bbox'][1],
                x2=text_elements[-1]['bbox'][2], y2=text_elements[-1]['bbox'][3],
                page=1
            )
            sections.append(Section(
                title="Email Body",
                content=body_content,
                level=1,
                bbox=bbox
            ))
        
        # Create attachments section
        if attachments:
            attachment_content = '\n'.join(e['content'] for e in attachments)
            bbox = BoundingBox(
                x1=attachments[0]['bbox'][0], y1=attachments[0]['bbox'][1],
                x2=attachments[-1]['bbox'][2], y2=attachments[-1]['bbox'][3],
                page=1
            )
            sections.append(Section(
                title="Attachments",
                content=attachment_content,
                level=1,
                bbox=bbox
            ))
        
        return sections
    
    def extract_key_values(self, headers: Dict[str, str], body: Dict[str, str]) -> List[KeyValue]:
        """Extract key-value pairs from email content."""
        key_values = []
        
        # Email headers as key-values
        header_mapping = {
            'from': 'sender',
            'to': 'recipient',
            'subject': 'subject',
            'date': 'date_sent'
        }
        
        for header, key_name in header_mapping.items():
            if header in headers:
                kv = KeyValue(
                    key=key_name,
                    value=headers[header],
                    confidence=1.0,
                    extraction_method="rule"
                )
                key_values.append(kv)
        
        # Extract potential key-values from body using simple patterns
        body_text = body.get('text', '')
        if body_text:
            # Look for patterns like "Key: Value"
            pattern = r'^([A-Za-z\s]+):\s*(.+)$'
            for line in body_text.split('\n'):
                match = re.match(pattern, line.strip())
                if match:
                    key, value = match.groups()
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    
                    if len(key) < 50 and len(value) < 200:  # Reasonable limits
                        kv = KeyValue(
                            key=f"email_{key}",
                            value=value,
                            confidence=0.7,
                            extraction_method="rule"
                        )
                        key_values.append(kv)
        
        return key_values
    
    def is_email_file(self, filename: str) -> bool:
        """Check if file is an email file."""
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        return extension in ['eml', 'msg', 'mbox']