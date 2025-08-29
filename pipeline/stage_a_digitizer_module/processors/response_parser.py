"""Response parsing utilities for CRF extraction"""

import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class ResponseParser:
    """Handles parsing of model responses"""
    
    def parse_json_lines(self, response: str, current_seq: int) -> List[Dict]:
        """
        Parse JSON Lines response into list of dicts with enhanced error handling
        """
        items = []
        
        # Check for special responses
        if "SH_NOT_FOUND" in response:
            logger.debug("No section headers found on this page")
            return []
        
        # Clean the response - remove any explanatory text before/after JSON
        lines = response.strip().split('\n')
        json_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line looks like JSON (starts with { and ends with })
            if line.startswith('{') and line.endswith('}'):
                json_lines.append(line)
            # Also try to extract JSON from lines that have JSON embedded
            elif '{' in line and '}' in line:
                try:
                    # Extract JSON part
                    start = line.index('{')
                    end = line.rindex('}') + 1
                    json_part = line[start:end]
                    json_lines.append(json_part)
                except ValueError:
                    continue
        
        # Parse each JSON line
        for line in json_lines:
            try:
                obj = json.loads(line)
                
                # Validate required fields based on tag
                tag = obj.get('tag')
                if tag == '<Q>':
                    if 'qid' not in obj:
                        # Use temporary ID for now - will be assigned properly later
                        obj['temp_qid'] = f"temp_q_{len([i for i in items if i.get('tag') == '<Q>'])}"
                        logger.debug(f"Question missing qid, assigned temp ID: {obj.get('text', '')[:50]}...")
                elif tag == '<INPUT>':
                    if 'parent_qid' not in obj:
                        # Use temporary parent ID for now
                        obj['parent_temp_qid'] = 'orphan'
                        logger.debug(f"Input missing parent_qid: {obj.get('text', '')[:50]}...")
                    if 'input_type' not in obj:
                        logger.warning(f"Input missing input_type: {obj.get('text', '')[:50]}...")
                        continue
                
                items.append(obj)
                
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON line: {line[:100]}... Error: {e}")
                continue
        
        # Log parsing results
        if items:
            tags_count = {}
            for item in items:
                tag = item.get('tag', 'unknown')
                tags_count[tag] = tags_count.get(tag, 0) + 1
            logger.debug(f"Parsed {len(items)} items: {tags_count}")
        else:
            logger.debug("No valid JSON items parsed from response")
        
        return items
    
    def parse_simple_tags(self, response: str, page_num: int, tag_type: str, 
                         current_seq: int) -> List[Dict]:
        """
        Parse simple tagged output from model
        
        Args:
            response: Model output with simple tags
            page_num: Current page number
            tag_type: Type of extraction ('questions', 'sections', 'form')
            current_seq: Current sequence counter
            
        Returns:
            List of parsed items as dicts
        """
        items = []
        lines = response.strip().split('\n')
        
        # Track question IDs for linking inputs - will be assigned later after sorting
        temp_qid_counter = 0
        temp_qid_map = {}  # Maps temporary IDs to items
        seq = current_seq
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for special responses
            if tag_type == 'sections' and line == 'SH_NOT_FOUND':
                logger.debug("No section headers found on this page")
                return []
            elif tag_type == 'form' and line == 'FORM_NOT_FOUND':
                logger.debug("No form name found on this page")
                return []
                
            # Parse tagged line
            if line.startswith('<Q>'):
                text = line[3:].strip()
                # Use temporary ID that will be replaced later
                temp_qid = f"temp_q_{temp_qid_counter}"
                temp_qid_counter += 1
                
                item = {
                    'seq': seq,
                    'tag': '<Q>',
                    'text': text,
                    'page': page_num + 1,
                    'temp_qid': temp_qid  # Temporary ID
                }
                seq += 1
                items.append(item)
                temp_qid_map[temp_qid] = item
                
            elif line.startswith('<INPUT>'):
                text = line[7:].strip()
                
                # Determine input type based on content
                input_type = 'text'  # default
                text_lower = text.lower()
                
                if text_lower in ['yes', 'no', 'n/a', 'unknown', 'not done']:
                    input_type = 'option'
                elif any(marker in text_lower for marker in ['___', 'dd-mmm-yyyy', 'hh:mm', '/___/', '|__|']):
                    input_type = 'text'
                elif len(text) > 50:  # Long text likely free form
                    input_type = 'free'
                elif text.startswith('□') or text.startswith('○'):
                    input_type = 'option'
                    
                # Get the most recent question's temp ID
                recent_temp_qid = None
                for i in range(len(items) - 1, -1, -1):
                    if items[i].get('tag') == '<Q>':
                        recent_temp_qid = items[i].get('temp_qid')
                        break
                    
                item = {
                    'seq': seq,
                    'tag': '<INPUT>',
                    'text': text,
                    'page': page_num + 1,
                    'parent_temp_qid': recent_temp_qid or 'orphan',  # Temporary parent ID
                    'input_type': input_type
                }
                seq += 1
                items.append(item)
                
            elif line.startswith('<SH>'):
                # Parse section header with optional question below
                content = line[4:].strip()
                
                # Check if there's a pipe separator indicating question below
                if '|' in content:
                    parts = content.split('|', 1)
                    text = parts[0].strip()
                    question_below = parts[1].strip()
                else:
                    text = content
                    question_below = "none"
                
                item = {
                    'seq': seq,
                    'tag': '<SH>',
                    'text': text,
                    'page': page_num + 1,
                    'question_below_text': question_below
                }
                seq += 1
                items.append(item)
                
            elif line.startswith('<FORM>'):
                text = line[6:].strip()
                item = {
                    'seq': seq,
                    'tag': '<FORM>',
                    'text': text,
                    'page': page_num + 1
                }
                seq += 1
                items.append(item)
                
        # Log parsing results
        if items:
            tags_count = {}
            for item in items:
                tag = item.get('tag', 'unknown')
                tags_count[tag] = tags_count.get(tag, 0) + 1
            logger.debug(f"Parsed {len(items)} items: {tags_count}")
        
        return items