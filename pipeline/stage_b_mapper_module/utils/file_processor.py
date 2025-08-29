"""File processing utilities for CRF JSON files"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FileProcessor:
    """Handles processing of CRF JSON files"""
    
    def process_json_file(self, json_path: Path, mapper) -> Dict[str, Any]:
        """Process a single CRF JSON file"""
        json_path = Path(json_path)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        results = {
            "file": str(json_path),
            "page_id": data.get("page_id", 0),
            "timestamp": datetime.now().isoformat(),
            "annotations": []
        }
        
        # Process items
        items = data.get("items", [])
        
        # Group by question
        current_question = None
        current_inputs = []
        
        for item in items:
            tag = item.get("tag")
            
            if tag == "<Q>":
                # Process previous question if exists
                if current_question:
                    field_data = self._build_field_data_from_crf(current_question, current_inputs)
                    annotation_result = mapper.annotate_field(field_data)
                    
                    results["annotations"].append({
                        "question": current_question,
                        "inputs": current_inputs,
                        "annotation": annotation_result
                    })
                
                # Start new question
                current_question = item
                current_inputs = []
                
            elif tag == "<INPUT>":
                current_inputs.append(item)
        
        # Process last question
        if current_question:
            field_data = self._build_field_data_from_crf(current_question, current_inputs)
            annotation_result = mapper.annotate_field(field_data)
            
            results["annotations"].append({
                "question": current_question,
                "inputs": current_inputs,
                "annotation": annotation_result
            })
        
        # Summary statistics
        results["summary"] = {
            "total_questions": len([a for a in results["annotations"]]),
            "annotated": len([a for a in results["annotations"] if a["annotation"].get("annotation") != "skip"]),
            "skipped": len([a for a in results["annotations"] if a["annotation"].get("annotation") == "skip"]),
            "valid": len([a for a in results["annotations"] if a["annotation"].get("valid", False)])
        }
        
        return results
    
    def process_json_directory(self, directory: Path, output_dir: Optional[Path], 
                             mapper) -> Dict[str, Any]:
        """Process directory of CRF JSON files"""
        directory = Path(directory)
        
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        # Find all page JSON files
        json_files = sorted(directory.glob("page_*.json"))
        
        if not json_files:
            logger.warning(f"No page JSON files found in {directory}")
            return {"error": "No files found"}
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        # Process each file
        all_results = {
            "directory": str(directory),
            "timestamp": datetime.now().isoformat(),
            "pages": []
        }
        
        for json_file in json_files:
            logger.info(f"Processing {json_file.name}")
            try:
                page_results = self.process_json_file(json_file, mapper)
                all_results["pages"].append(page_results)
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                all_results["pages"].append({
                    "file": str(json_file),
                    "error": str(e)
                })
        
        # Generate summary
        all_results["summary"] = self._generate_directory_summary(all_results["pages"])
        
        # Save results if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{directory.name}_annotations.json"
            
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
                
            logger.info(f"Results saved to {output_file}")
        
        return all_results
    
    def _build_field_data_from_crf(self, question: Dict, inputs: List[Dict]) -> Dict[str, Any]:
        """Build field_data from CRF question and inputs"""
        field_data = {
            "label": question.get("text", ""),
            "type": "unknown",
            "options": [],
            "metadata": {
                "qid": question.get("qid"),
                "page": question.get("page"),
                "section": question.get("section"),
                "form": question.get("form"),
                "bbox": question.get("bbox_xyxy")
            }
        }
        
        # Determine type and extract options
        if not inputs:
            field_data["type"] = "text"
        else:
            # Analyze input types
            input_types = [inp.get("input_type", "") for inp in inputs]
            input_texts = [inp.get("text", "") for inp in inputs]
            
            if all(t == "option" for t in input_types):
                field_data["type"] = "radio"
                field_data["options"] = input_texts
            elif any("checkbox" in t for t in input_types):
                field_data["type"] = "checkbox"
                field_data["options"] = input_texts
            elif len(inputs) == 1 and inputs[0].get("input_type") == "text":
                field_data["type"] = "text"
                # Check for specific text patterns
                text = inputs[0].get("text", "").lower()
                if "date" in text or "dd-mmm-yyyy" in text:
                    field_data["type"] = "date"
                elif any(marker in text for marker in ["___", "|__|", "____"]):
                    field_data["type"] = "number" if any(c.isdigit() for c in text) else "text"
            else:
                # Mixed types
                field_data["type"] = "mixed"
                field_data["options"] = input_texts
        
        return field_data
    
    def _generate_directory_summary(self, page_results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for directory processing"""
        total_questions = 0
        total_annotated = 0
        total_skipped = 0
        total_valid = 0
        domain_counts = {}
        pattern_counts = {}
        
        for page in page_results:
            if "error" not in page and "summary" in page:
                summary = page["summary"]
                total_questions += summary.get("total_questions", 0)
                total_annotated += summary.get("annotated", 0)
                total_skipped += summary.get("skipped", 0)
                total_valid += summary.get("valid", 0)
                
                # Count domains and patterns
                for annotation in page.get("annotations", []):
                    result = annotation.get("annotation", {})
                    domain = result.get("domain", "UNKNOWN")
                    pattern = result.get("pattern", "unknown")
                    
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        return {
            "total_pages": len(page_results),
            "total_questions": total_questions,
            "total_annotated": total_annotated,
            "total_skipped": total_skipped,
            "total_valid": total_valid,
            "domains": domain_counts,
            "patterns": pattern_counts
        }