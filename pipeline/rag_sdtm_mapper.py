#!/usr/bin/env python3
"""
RAG-based SDTM Mapper with Constrained Selection and Abstention
Ensures high accuracy by grounding all decisions in knowledge base retrieval
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval with score and metadata"""
    item_id: str
    score: float
    content: Dict[str, Any]
    source: str = ""


@dataclass
class PatternFeatures:
    """Extracted features for pattern detection"""
    has_units: bool = False
    units: List[str] = field(default_factory=list)
    has_criterion_code: bool = False
    criterion_codes: List[str] = field(default_factory=list)
    has_other_specify: bool = False
    has_date_pattern: bool = False
    has_variable_hint: bool = False
    variable_hints: List[str] = field(default_factory=list)
    is_operational: bool = False
    measurement_indicators: List[str] = field(default_factory=list)


@dataclass
class MappingDecision:
    """Final mapping decision with confidence and abstention support"""
    pattern: str
    domain: str
    variables: List[str]
    values: Dict[str, str]
    annotation: str
    confidence: float
    abstain: bool = False
    abstain_reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


class SDTMRetriever:
    """Multi-index retrieval system for SDTM knowledge base"""
    
    def __init__(self, kb_dir: Path):
        self.kb_dir = Path(kb_dir)
        
        # Indexes
        self.domain_index = {}
        self.variable_index = defaultdict(dict)  # domain -> {var -> info}
        self.ct_index = defaultdict(list)  # codelist -> terms
        
        # TF-IDF vectorizers
        self.domain_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.variable_vectorizers = {}  # per domain
        
        # Dense embeddings (lightweight model)
        self.encoder = None
        self.domain_embeddings = None
        self.variable_embeddings = {}
        
        # Domain keywords for rule-based boosting
        self.domain_keywords = {
            "DM": ["demographics", "age", "sex", "race", "ethnicity", "birth", "gender"],
            "VS": ["vital signs", "blood pressure", "temperature", "pulse", "heart rate", "weight", "height", "respiration"],
            "AE": ["adverse event", "side effect", "toxicity", "seriousness", "severity", "action taken"],
            "CM": ["concomitant medication", "medication", "drug", "dose", "route", "frequency"],
            "MH": ["medical history", "past medical", "diagnosis", "condition"],
            "LB": ["laboratory", "lab", "chemistry", "hematology", "urinalysis", "serum", "blood"],
            "EG": ["ecg", "electrocardiogram", "qt", "pr interval", "qrs"],
            "IE": ["inclusion", "exclusion", "eligibility", "criteria", "criterion"],
            "SV": ["visit", "visit date", "study day"],
            "EX": ["exposure", "study drug", "dosing", "administration"],
            "DS": ["disposition", "completion", "discontinuation", "withdrawal"],
            "PE": ["physical exam", "examination", "abnormal", "normal"],
            "QS": ["questionnaire", "scale", "score", "assessment"],
            "SC": ["subject characteristics", "baseline characteristics"],
            "FA": ["findings about", "tumor", "lesion"]
        }
        
    def load(self):
        """Load and index all knowledge base components"""
        self._load_domains()
        self._load_variables()
        self._load_controlled_terms()
        self._build_indexes()
        
    def _load_domains(self):
        """Load domain definitions"""
        domains_file = self.kb_dir / "domains.json"
        if domains_file.exists():
            with open(domains_file, 'r') as f:
                domains = json.load(f)
            for domain in domains:
                self.domain_index[domain["code"]] = domain
        logger.info(f"Loaded {len(self.domain_index)} domains")
        
    def _load_variables(self):
        """Load variables organized by domain"""
        variables_file = self.kb_dir / "variables_all.json"
        if variables_file.exists():
            with open(variables_file, 'r') as f:
                variables = json.load(f)
            for var in variables:
                domain = var.get("domain", "")
                if domain:
                    self.variable_index[domain][var["name"]] = var
        logger.info(f"Loaded variables for {len(self.variable_index)} domains")
        
    def _load_controlled_terms(self):
        """Load controlled terminology"""
        ct_file = self.kb_dir / "cdisc_ct.json"
        if ct_file.exists():
            with open(ct_file, 'r') as f:
                ct_data = json.load(f)
            
            if "codelists" in ct_data:
                for codelist_item in ct_data["codelists"]:
                    codelist_info = codelist_item.get("codelist", {})
                    codelist_name = codelist_info.get("shortName", "")
                    
                    if codelist_name:
                        for term in codelist_item.get("terms", []):
                            self.ct_index[codelist_name].append({
                                "code": term.get("submissionValue", ""),
                                "name": term.get("preferredTerm", ""),
                                "definition": term.get("definition", ""),
                                "synonyms": term.get("synonyms", [])
                            })
        logger.info(f"Loaded {len(self.ct_index)} codelists")
        
    def _build_indexes(self):
        """Build TF-IDF indexes for retrieval"""
        # Domain index
        domain_texts = []
        domain_codes = []
        for code, info in self.domain_index.items():
            # Combine name, description, and keywords
            text = f"{info['name']} {info['description']}"
            if code in self.domain_keywords:
                text += " " + " ".join(self.domain_keywords[code])
            domain_texts.append(text)
            domain_codes.append(code)
            
        if domain_texts:
            self.domain_tfidf = self.domain_vectorizer.fit_transform(domain_texts)
            self.domain_codes = domain_codes
            
        # Variable indexes per domain
        for domain, variables in self.variable_index.items():
            var_texts = []
            var_names = []
            for name, info in variables.items():
                text = f"{name} {info.get('label', '')} {info.get('description', '')}"
                var_texts.append(text)
                var_names.append(name)
                
            if var_texts:
                vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
                self.variable_vectorizers[domain] = {
                    'vectorizer': vectorizer,
                    'tfidf': vectorizer.fit_transform(var_texts),
                    'names': var_names
                }
                
    def retrieve_domains(self, query: str, form: str = "", section: str = "", k: int = 10) -> List[RetrievalResult]:
        """Retrieve top-k domain candidates"""
        # Combine query with context
        full_query = f"{query} {form} {section}".lower()
        
        # TF-IDF similarity
        query_vec = self.domain_vectorizer.transform([full_query])
        similarities = cosine_similarity(query_vec, self.domain_tfidf).flatten()
        
        # Rule-based boosting
        boosts = np.zeros(len(self.domain_codes))
        for i, code in enumerate(self.domain_codes):
            # Form/section exact match boost
            domain_info = self.domain_index[code]
            if domain_info['name'].lower() in form.lower():
                boosts[i] += 0.3
            if domain_info['name'].lower() in section.lower():
                boosts[i] += 0.2
                
            # Keyword matching boost
            if code in self.domain_keywords:
                for keyword in self.domain_keywords[code]:
                    if keyword in full_query:
                        boosts[i] += 0.1
                        
        # Combined scores
        final_scores = 0.7 * similarities + 0.3 * boosts
        
        # Get top-k
        top_indices = np.argsort(final_scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            code = self.domain_codes[idx]
            results.append(RetrievalResult(
                item_id=code,
                score=float(final_scores[idx]),
                content=self.domain_index[code],
                source="domain_index"
            ))
            
        return results
        
    def retrieve_variables(self, query: str, domain: str, pattern: str = "", k: int = 15) -> List[RetrievalResult]:
        """Retrieve variables for a specific domain"""
        if domain not in self.variable_vectorizers:
            return []
            
        vectorizer_data = self.variable_vectorizers[domain]
        query_vec = vectorizer_data['vectorizer'].transform([query])
        similarities = cosine_similarity(query_vec, vectorizer_data['tfidf']).flatten()
        
        # Pattern-based filtering
        role_filter = np.ones(len(vectorizer_data['names']))
        if pattern == "FINDINGS":
            # Prioritize ORRES, ORRESU, TESTCD variables
            for i, name in enumerate(vectorizer_data['names']):
                if name.endswith(('ORRES', 'ORRESU', 'TESTCD')):
                    role_filter[i] = 2.0
                else:
                    role_filter[i] = 0.1
                    
        # Combined scores
        final_scores = similarities * role_filter
        
        # Get top-k
        top_indices = np.argsort(final_scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if final_scores[idx] > 0:
                var_name = vectorizer_data['names'][idx]
                results.append(RetrievalResult(
                    item_id=var_name,
                    score=float(final_scores[idx]),
                    content=self.variable_index[domain][var_name],
                    source=f"variable_index_{domain}"
                ))
                
        return results
        
    def retrieve_testcodes(self, query: str, domain: str, k: int = 10) -> List[RetrievalResult]:
        """Retrieve test codes for findings domains"""
        testcd_var = f"{domain}TESTCD"
        if testcd_var not in self.ct_index:
            return []
            
        # Simple text matching for test codes
        results = []
        query_lower = query.lower()
        
        for term in self.ct_index[testcd_var]:
            code = term['code']
            name = term['name'].lower()
            synonyms = [s.lower() for s in term.get('synonyms', [])]
            
            # Score based on matches
            score = 0.0
            if code.lower() in query_lower:
                score += 1.0
            if any(word in name for word in query_lower.split()):
                score += 0.8
            if any(word in syn for syn in synonyms for word in query_lower.split()):
                score += 0.6
                
            if score > 0:
                results.append(RetrievalResult(
                    item_id=code,
                    score=score,
                    content=term,
                    source=f"ct_index_{testcd_var}"
                ))
                
        # Sort by score and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]


class FeatureExtractor:
    """Extract features from questions for pattern detection"""
    
    # Unit patterns
    UNIT_PATTERNS = [
        r'\(([^)]+)\)',  # Anything in parentheses
        r'\b(mg|g|kg|lb|cm|m|mm|°C|°F|mmHg|bpm|/min|mL|L|%)\b'
    ]
    
    # Criterion code patterns
    CRITERION_PATTERNS = [
        r'\(INC\d+\)',
        r'\(EX\d+\)'
    ]
    
    # Date patterns
    DATE_PATTERNS = [
        r'dd-mmm-yyyy',
        r'yyyy-mm-dd',
        r'date',
        r'datetime'
    ]
    
    # Operational indicators
    OPERATIONAL_INDICATORS = [
        'page number', 'crf page', 'monitor', 'signature', 
        'initials', 'data entry', 'not applicable', 'n/a'
    ]
    
    # Measurement indicators
    MEASUREMENT_INDICATORS = [
        'blood pressure', 'temperature', 'pulse', 'heart rate',
        'weight', 'height', 'lab', 'laboratory', 'hemoglobin',
        'glucose', 'cholesterol', 'systolic', 'diastolic'
    ]
    
    @classmethod
    def extract(cls, question: Dict, context: Dict) -> PatternFeatures:
        """Extract features from question and context"""
        features = PatternFeatures()
        
        text = question.get('text', '').lower()
        original_text = question.get('text', '')
        
        # Extract units
        for pattern in cls.UNIT_PATTERNS:
            matches = re.findall(pattern, original_text)
            if matches:
                features.has_units = True
                features.units.extend(matches)
                
        # Extract criterion codes
        for pattern in cls.CRITERION_PATTERNS:
            matches = re.findall(pattern, original_text)
            if matches:
                features.has_criterion_code = True
                features.criterion_codes.extend(matches)
                
        # Check for "other, specify"
        if 'other' in text and 'specify' in text:
            features.has_other_specify = True
            
        # Check for date patterns
        for pattern in cls.DATE_PATTERNS:
            if pattern in text:
                features.has_date_pattern = True
                break
                
        # Extract variable hints (e.g., (VSTDT))
        var_hint_matches = re.findall(r'\(([A-Z]{2,}[A-Z0-9_]*)\)', original_text)
        if var_hint_matches:
            features.has_variable_hint = True
            features.variable_hints = var_hint_matches
            
        # Check if operational
        for indicator in cls.OPERATIONAL_INDICATORS:
            if indicator in text:
                features.is_operational = True
                break
                
        # Check for measurements
        for indicator in cls.MEASUREMENT_INDICATORS:
            if indicator in text:
                features.measurement_indicators.append(indicator)
                
        return features


class ConstrainedSDTMMapper:
    """SDTM Mapper with constrained selection and abstention"""
    
    def __init__(self, kb_dir: Path, model_path: str = "Qwen/Qwen2.5-14B-Instruct"):
        self.kb_dir = Path(kb_dir)
        self.model_path = model_path
        
        # Initialize components
        self.retriever = SDTMRetriever(kb_dir)
        self.retriever.load()
        
        # Confidence thresholds
        self.domain_confidence_threshold = 0.15  # Margin between top 2
        self.variable_confidence_threshold = 0.1
        self.pattern_confidence_threshold = 0.7
        
        # Initialize model
        self._init_model()
        
    def _init_model(self):
        """Initialize the LLM for constrained selection"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
    def _query_llm_json(self, prompt: str, max_tokens: int = 256) -> Optional[Dict]:
        """Query LLM and parse JSON response"""
        messages = [
            {"role": "system", "content": "You are an SDTM mapping expert. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
            
        return None
        
    def detect_pattern(self, question: Dict, context: Dict, features: PatternFeatures) -> Tuple[str, float]:
        """Detect pattern using rules and optionally verify with LLM"""
        
        # Rule-based detection
        if features.has_criterion_code or (
            context.get('section', '').lower() in ['inclusion criteria', 'exclusion criteria', 'eligibility']
        ):
            return "CONDITIONAL", 0.95
            
        if features.has_other_specify:
            return "SUPPLEMENTAL", 0.95
            
        if features.is_operational:
            return "NOT_SUBMITTED", 0.95
            
        if features.has_units or features.measurement_indicators:
            return "FINDINGS", 0.9
            
        if features.has_date_pattern:
            return "PLAIN", 0.85
            
        # LLM verification for uncertain cases
        prompt = f"""Task: Choose ONE pattern label or ABSTAIN.

Question: "{question['text']}"
Form: "{context.get('form_name', '')}"
Section: "{context.get('section', '')}"

Signals:
- Units found: {features.has_units}, {features.units}
- Criterion code: {features.criterion_codes or 'none'}
- 'Other, specify' phrase: {features.has_other_specify}
- Date pattern: {features.has_date_pattern}
- Measurement indicators: {features.measurement_indicators}

Allowed patterns:
- PLAIN: Direct variable mapping
- FINDINGS: Measurements with test codes
- CONDITIONAL: When/then conditions (includes IE criteria)
- SUPPLEMENTAL: "Other, specify" fields
- NOT_SUBMITTED: Operational/administrative

Answer as JSON: {{"pattern": "<PATTERN>", "confidence": 0.0-1.0}}"""

        result = self._query_llm_json(prompt, max_tokens=100)
        
        if result and 'pattern' in result:
            pattern = result['pattern']
            confidence = result.get('confidence', 0.7)
            
            # Validate pattern
            valid_patterns = ["PLAIN", "FINDINGS", "CONDITIONAL", "SUPPLEMENTAL", "NOT_SUBMITTED"]
            if pattern in valid_patterns:
                return pattern, confidence
                
        # Default to PLAIN with low confidence
        return "PLAIN", 0.5
        
    def select_domain(self, question: Dict, context: Dict, pattern: str, features: PatternFeatures) -> Tuple[Optional[str], float, List[RetrievalResult]]:
        """Select domain using retrieval and constrained LLM selection"""
        
        # Retrieve top domains
        query = question['text']
        form = context.get('form_name', '')
        section = context.get('section', '')
        
        # Pattern-specific priors
        if pattern == "CONDITIONAL" and features.has_criterion_code:
            # If we have INC/EX codes, strongly suggest IE domain
            ie_result = RetrievalResult(
                item_id="IE",
                score=0.9,
                content=self.retriever.domain_index.get("IE", {}),
                source="pattern_prior"
            )
            # Still retrieve others but IE will be top
            domain_results = self.retriever.retrieve_domains(query, form, section, k=9)
            return "IE", 0.9, [ie_result] + domain_results
            
        # Retrieve candidates
        domain_results = self.retriever.retrieve_domains(query, form, section, k=10)
        
        if not domain_results:
            return None, 0.0, []
            
        # Check margin between top 2
        if len(domain_results) >= 2:
            margin = domain_results[0].score - domain_results[1].score
            if margin < self.domain_confidence_threshold:
                # Low confidence, use LLM to decide
                candidates_text = []
                for r in domain_results[:5]:
                    dom = r.content
                    candidates_text.append(f"- {r.item_id}: {dom['name']} - {dom['description'][:100]}...")
                    
                prompt = f"""Task: Pick exactly one domain code from this list or NONE.

Question: "{query}"
Form/Section: "{form} | {section}"
Pattern: {pattern}

Top candidates:
{chr(10).join(candidates_text)}

Rules:
- Choose the best-supported domain based on the question content
- If uncertain, return NONE

Return JSON: {{"domain": "XX" or "NONE", "reason": "brief explanation"}}"""

                result = self._query_llm_json(prompt, max_tokens=150)
                
                if result and result.get('domain') != 'NONE':
                    selected = result['domain']
                    # Verify it's in our candidates
                    if any(r.item_id == selected for r in domain_results[:5]):
                        return selected, 0.8, domain_results
                        
                # Abstain if uncertain
                return None, margin, domain_results
                
        # High confidence in top result
        return domain_results[0].item_id, domain_results[0].score, domain_results
        
    def select_variables(self, question: Dict, domain: str, pattern: str, features: PatternFeatures) -> Tuple[List[str], Dict[str, str], float]:
        """Select variables based on pattern and domain"""
        
        variables = []
        values = {}
        confidence = 0.0
        
        if pattern == "NOT_SUBMITTED":
            return [], {}, 1.0
            
        elif pattern == "FINDINGS":
            # For findings, we need ORRES, potentially ORRESU, and TESTCD
            base_vars = [f"{domain}ORRES"]
            if features.units:
                base_vars.append(f"{domain}ORRESU")
                
            # Find appropriate test code
            testcd_var = f"{domain}TESTCD"
            testcode_results = self.retriever.retrieve_testcodes(question['text'], domain, k=5)
            
            if testcode_results:
                # Use LLM to select from candidates
                candidates_text = []
                for r in testcode_results:
                    term = r.content
                    candidates_text.append(f"- {term['code']}: {term['name']}")
                    
                prompt = f"""Task: Choose ONE TESTCD for the question from this list or NONE.

Question: "{question['text']}"
Domain: {domain}

Allowed TESTCDs:
{chr(10).join(candidates_text)}

Return JSON: {{"testcd": "<CODE>" or "NONE"}}"""

                result = self._query_llm_json(prompt, max_tokens=100)
                
                if result and result.get('testcd') != 'NONE':
                    values[testcd_var] = result['testcd']
                    confidence = 0.9
                else:
                    # Try to generate a reasonable code
                    values[testcd_var] = self._generate_testcode(question['text'])
                    confidence = 0.6
            else:
                values[testcd_var] = "UNKNOWN"
                confidence = 0.3
                
            return base_vars, values, confidence
            
        elif pattern == "SUPPLEMENTAL":
            # Generate QNAM
            qnam = self._generate_qnam(question['text'])
            values['QNAM'] = qnam
            values['SUPP_DOMAIN'] = f"SUPP{domain}"
            return [], values, 0.8
            
        elif pattern == "CONDITIONAL":
            # CONDITIONAL pattern can be used for various when/then scenarios
            # Check if this is IE domain based on features
            if domain == "IE" and features.criterion_codes:
                # IE domain criteria
                variables = ["IEORRES"]
                
                # Extract criterion code
                code = features.criterion_codes[0].strip('()')
                values['IETESTCD'] = code
                confidence = 0.95
                
                return variables, values, confidence
            elif domain == "IE":
                # IE without code - need to extract or abstain
                variables = ["IEORRES"]
                values['IETESTCD'] = "UNKNOWN"
                confidence = 0.3
                
                return variables, values, confidence
            else:
                # Other domains using when/then pattern
                # This would need domain-specific logic
                # For now, treat as PLAIN with lower confidence
                return self.select_variables(question, domain, "PLAIN", features)
            
        else:  # PLAIN
            # Retrieve candidate variables
            var_results = self.retriever.retrieve_variables(question['text'], domain, pattern, k=10)
            
            if not var_results:
                return [], {}, 0.0
                
            # Use LLM to select
            candidates_text = []
            for r in var_results[:7]:
                var = r.content
                candidates_text.append(f"- {var['name']}: {var['label']} ({var.get('type', 'Char')})")
                
            prompt = f"""Task: Choose ONE or TWO variables from this list or NONE.

Domain: {domain}
Question: "{question['text']}"

Candidates:
{chr(10).join(candidates_text)}

Rules:
- Choose the most appropriate variable(s)
- Use / to separate multiple variables
- Common mappings: DOB→BRTHDTC, Gender→SEX, Race→RACE

Return JSON: {{"variables": ["VAR1"] or ["VAR1", "VAR2"] or []}}"""

            result = self._query_llm_json(prompt, max_tokens=150)
            
            if result and result.get('variables'):
                variables = result['variables']
                # Validate they're in our candidates
                valid_vars = [r.item_id for r in var_results]
                variables = [v for v in variables if v in valid_vars]
                
                if variables:
                    confidence = 0.8
                else:
                    confidence = 0.0
            else:
                confidence = 0.0
                
            return variables, {}, confidence
            
    def format_annotation(self, pattern: str, domain: str, variables: List[str], values: Dict[str, str]) -> str:
        """Format annotation according to SDTM-MSG Section 3"""
        
        if pattern == "NOT_SUBMITTED":
            return "[NOT SUBMITTED]"
            
        elif pattern == "FINDINGS":
            # Format: VSORRES [/ VSORRESU] when VSTESTCD = CODE
            base = " / ".join(variables)
            testcd_parts = []
            for k, v in values.items():
                if k.endswith("TESTCD"):
                    testcd_parts.append(f"{k} = {v}")
            
            if testcd_parts:
                return f"{base} when {' and '.join(testcd_parts)}"
            return base
            
        elif pattern == "SUPPLEMENTAL":
            # Format: QNAM in SUPPDM
            if 'QNAM' in values and 'SUPP_DOMAIN' in values:
                return f"{values['QNAM']} in {values['SUPP_DOMAIN']}"
            return "UNKNOWN in SUPPUNK"
            
        elif pattern == "CONDITIONAL":
            # Format: <var> when <var> = <value>
            # Most common case is IE domain
            if 'IETESTCD' in values:
                return f"IEORRES when IETESTCD = {values['IETESTCD']}"
            elif values:
                # Generic when/then format
                conditions = []
                for k, v in values.items():
                    if not k.startswith('SUPP'):
                        conditions.append(f"{k} = {v}")
                if conditions and variables:
                    return f"{' / '.join(variables)} when {' and '.join(conditions)}"
            return "UNKNOWN when UNKNOWN = UNKNOWN"
            
        else:  # PLAIN
            # Format: VAR or VAR1 / VAR2
            if variables:
                return " / ".join(variables)
            return "UNKNOWN"
            
    def validate_mapping(self, decision: MappingDecision) -> Tuple[bool, List[str]]:
        """Validate the mapping decision"""
        errors = []
        
        # Check domain exists
        if decision.domain not in self.retriever.domain_index:
            errors.append(f"Invalid domain: {decision.domain}")
            
        # Check variables exist
        for var in decision.variables:
            if var not in self.retriever.variable_index.get(decision.domain, {}):
                errors.append(f"Invalid variable: {var} in domain {decision.domain}")
                
        # Check controlled terms
        for var, value in decision.values.items():
            if var.endswith("TESTCD") and var in self.retriever.ct_index:
                valid_codes = [term['code'] for term in self.retriever.ct_index[var]]
                if value not in valid_codes and value != "UNKNOWN":
                    errors.append(f"Invalid {var} value: {value}")
                    
        # Format validation
        pattern_regexes = {
            "PLAIN": r"^[A-Z][A-Z0-9]{0,7}(?:\s*/\s*[A-Z][A-Z0-9]{0,7})*$",
            "FINDINGS": r"^[A-Z]{2}ORRES(?:\s*/\s*[A-Z]{2}ORRESU)?\s+when\s+[A-Z]{2}TESTCD\s*=\s*[A-Z0-9]{1,8}$",
            "SUPPLEMENTAL": r"^[A-Z0-9_]{1,8}\s+in\s+SUPP[A-Z]{2}$",
            "CONDITIONAL": r"^[A-Z][A-Z0-9]{0,7}(?:\s*/\s*[A-Z][A-Z0-9]{0,7})?\s+when\s+[A-Z][A-Z0-9]{0,7}\s*=\s*[A-Z0-9_]+$",
            "NOT_SUBMITTED": r"^\[NOT SUBMITTED\]$"
        }
        
        if decision.pattern in pattern_regexes:
            if not re.match(pattern_regexes[decision.pattern], decision.annotation):
                errors.append(f"Invalid format for pattern {decision.pattern}")
                
        return len(errors) == 0, errors
        
    def map_question(self, question: Dict, context: Dict) -> MappingDecision:
        """Main entry point for mapping a question"""
        
        # Extract features
        features = FeatureExtractor.extract(question, context)
        
        # Detect pattern
        pattern, pattern_confidence = self.detect_pattern(question, context, features)
        
        if pattern_confidence < self.pattern_confidence_threshold:
            return MappingDecision(
                pattern="UNKNOWN",
                domain="",
                variables=[],
                values={},
                annotation="",
                confidence=0.0,
                abstain=True,
                abstain_reason=f"Low pattern confidence: {pattern_confidence:.2f}"
            )
            
        # Select domain
        domain, domain_confidence, domain_results = self.select_domain(question, context, pattern, features)
        
        if not domain:
            return MappingDecision(
                pattern=pattern,
                domain="",
                variables=[],
                values={},
                annotation="",
                confidence=0.0,
                abstain=True,
                abstain_reason="Could not determine domain",
                evidence={"domain_candidates": [r.item_id for r in domain_results[:5]]}
            )
            
        # Select variables
        variables, values, var_confidence = self.select_variables(question, domain, pattern, features)
        
        # Format annotation
        annotation = self.format_annotation(pattern, domain, variables, values)
        
        # Create decision
        decision = MappingDecision(
            pattern=pattern,
            domain=domain,
            variables=variables,
            values=values,
            annotation=annotation,
            confidence=min(pattern_confidence, domain_confidence, var_confidence),
            evidence={
                "features": features.__dict__,
                "domain_score": domain_confidence,
                "variable_score": var_confidence
            }
        )
        
        # Validate
        is_valid, errors = self.validate_mapping(decision)
        
        if not is_valid:
            decision.abstain = True
            decision.abstain_reason = f"Validation errors: {'; '.join(errors)}"
            
        # Final confidence check
        if decision.confidence < 0.5:
            decision.abstain = True
            decision.abstain_reason = f"Low overall confidence: {decision.confidence:.2f}"
            
        return decision
        
    def _generate_qnam(self, text: str) -> str:
        """Generate QNAM for supplemental qualifiers"""
        # Remove common words and create abbreviation
        words = re.findall(r'\b[A-Z]+\b', text.upper())
        stop_words = {"THE", "A", "AN", "OF", "IN", "FOR", "TO", "AND", "OR", "OTHER", "SPECIFY"}
        words = [w for w in words if w not in stop_words and len(w) > 1]
        
        if words:
            # Take first letters of first few words
            qnam = ''.join(w[0] for w in words[:4])
            # Add suffix if needed to make unique
            if len(qnam) < 3:
                qnam += "OTH"
            return qnam[:8]  # Limit to 8 chars
        else:
            return "MISCOTH"
            
    def _generate_testcode(self, text: str) -> str:
        """Generate a test code from question text"""
        # Common mappings
        mappings = {
            "systolic": "SYSBP",
            "diastolic": "DIABP",
            "temperature": "TEMP",
            "weight": "WEIGHT",
            "height": "HEIGHT",
            "pulse": "PULSE",
            "heart rate": "HR"
        }
        
        text_lower = text.lower()
        for key, code in mappings.items():
            if key in text_lower:
                return code
                
        # Generate from words
        words = re.findall(r'\b[A-Z]+\b', text.upper())
        if words:
            return ''.join(w[0] for w in words[:4])[:8]
        return "TEST"


def process_page_with_rag(page_data: Dict, mapper: ConstrainedSDTMMapper) -> Dict:
    """Process a page using the RAG-based mapper"""
    
    # Extract context
    context = {
        'form_name': '',
        'section': '',
        'study_name': ''
    }
    
    # Get form name
    items = page_data.get('items', [])
    for item in items:
        if item.get('tag') == '<FORM>':
            context['form_name'] = item.get('text', '')
            break
            
    # Get sections
    sections = []
    for item in items:
        if item.get('tag') == '<SH>':
            sections.append(item.get('text', ''))
    
    # Process questions
    results = []
    questions = [item for item in items if item.get('tag') == '<Q>']
    
    for question in questions:
        # Add section context
        if sections:
            context['section'] = sections[0]  # Use first section for now
            
        # Map question
        decision = mapper.map_question(question, context)
        
        # Create result
        result = {
            'question_id': question.get('qid', ''),
            'question_text': question.get('text', ''),
            'pattern': decision.pattern,
            'domain': decision.domain,
            'variables': decision.variables,
            'values': decision.values,
            'annotation': decision.annotation,
            'confidence': decision.confidence,
            'abstained': decision.abstain,
            'abstain_reason': decision.abstain_reason,
            'evidence': decision.evidence
        }
        
        results.append(result)
        
    return {
        'page_id': page_data.get('page_id', 0),
        'context': context,
        'annotations': results,
        'summary': {
            'total_questions': len(questions),
            'mapped': sum(1 for r in results if not r['abstained']),
            'abstained': sum(1 for r in results if r['abstained'])
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG-based SDTM Mapper')
    parser.add_argument('input_file', help='Input JSON file (e.g., page_000.json)')
    parser.add_argument('--kb-dir', required=True, help='Knowledge base directory')
    parser.add_argument('--output', help='Output file')
    parser.add_argument('--model', default='Qwen/Qwen2.5-14B-Instruct', help='Model to use')
    
    args = parser.parse_args()
    
    # Initialize mapper
    mapper = ConstrainedSDTMMapper(
        kb_dir=Path(args.kb_dir),
        model_path=args.model
    )
    
    # Load input
    with open(args.input_file, 'r') as f:
        page_data = json.load(f)
        
    # Process
    results = process_page_with_rag(page_data, mapper)
    
    # Save output
    output_file = args.output or args.input_file.replace('.json', '_rag_mapped.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Print summary
    print(f"\nMapping Summary:")
    print(f"Total questions: {results['summary']['total_questions']}")
    print(f"Successfully mapped: {results['summary']['mapped']}")
    print(f"Abstained: {results['summary']['abstained']}")
    print(f"\nResults saved to: {output_file}")