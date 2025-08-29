# SDTM Annotation Instructions

## 0) Goal
Given CRF/eCRF text (form labels, question text, options, field hints), produce a machine‑readable list of annotations that maps each collected data item to SDTM domain/variable(s) and, when needed, explains relationships, origins, and special cases (supplemental qualifiers, RELREC, not‑submitted prompts, etc.). The output must enable a reviewer to trace where each SDTM variable comes from, even though no visual markup is available. (Purpose of aCRF and traceability per SDTM‑MSG §3.0–3.1. Basic principles emphasize searchable text and that aCRF reflects data intended for SDTM submission. Use Define‑XML "HasNoData" to indicate planned‑but‑empty datasets/variables.)

## 1) Input you will receive
- **Document metadata**: study id, version(s), optional visit schedule.
- **Form text**: page/form title, field labels, response choices, units, help text, "Other, specify" prompts, etc.
- **Optional**: a list of planned SDTM domains for the study, plus known codelists.

Do not assume visuals. Work only from text. Where the MSG uses fonts/colors/borders, simulate with tags in your output. (MSG appearance guidance in §3.1.2.)

## 2) Output format (strict JSON Lines)
Produce one JSON object per annotation (one line per object). Use this schema:

```json
{
  "form_id": "string",              // e.g., "VS v2.0", "DM", "Adverse Events"
  "page_version": "string|null",    // if provided (e.g., "v1.0", "v2.0")
  "visit_context": "string|null",   // e.g., "Week 12", or "Running Record"
  "text_anchor": "string",          // short snippet of the CRF text you mapped
  "domain": "string",               // SDTM domain code, e.g., "VS","DM","AE","QS"
  "dataset_name": "string|null",    // if domain is split, e.g., "QSPH","QSSL","LBCH"
  "variables": [                    // one or more SDTM variables; ALL CAPS
    "VSORRES",
    "VSORRESU"
  ],
  "when": "string|null",            // "<VAR(S)> when <TESTCD>=<VALUE>", e.g., "VSORRES/VSORRESU when VSTESTCD=TEMP"
  "origin": "string",               // "Collected","Derived","Predecessor","Assigned","ePRO Collected", etc.
  "collected_boolean": true,        // true if CRF-collected; false if only clarifying
  "not_submitted": false,           // true if prompt collected but not in SDTM (mark as [NOT SUBMITTED] in notes)
  "supplemental": {
    "is_supplemental": false,
    "qnam": "string|null",          // e.g., "RACEOTH"
    "supp_domain": "string|null"    // e.g., "SUPPDM"
  },
  "relrec": {
    "has_relrec": false,
    "statement": "string|null"      // e.g., "RELREC when DDLNKID = AE.AELNKID"
  },
  "repeat_number_var": "string|null", // e.g., "VSREPNUM","FTREPNUM" if repeated measures present
  "units_var": "string|null",       // e.g., "--ORRESU" if a unit is captured
  "codelist_hint": "string|null",   // name/summary of enumerations seen on the form
  "notes": "string|null"            // sentence-case comments for reviewers
}
```

Why: MSG asks domain names (not split dataset names) to be annotated and variables to be capitalized; multiple variables in one annotation separated with "/"; "when/then" format for Findings; mark not‑submitted prompts; and annotate QNAM in SUPPxx for supplemental qualifiers. (§3.1.2 and §3.1.3).

Also return two document‑level objects (once per form bundle):

```json
{
  "document_toc": {
    "by_visits": ["Screening 1","Screening 2","Week 2","Week 4","…","Running Records"],
    "by_forms":  ["Demographics","Vital Signs","Adverse Events","ConMeds","…"]
  },
  "replaced_or_deprecated_pages": [
    {"form_id":"VS","old_version":"v1.0","new_version":"v2.0","comment":"Triplicate BP/pulse introduced; keep both for traceability"}
  ]
}
```

Dual bookmarking (chronology & forms) and handling of replacement/deprecated pages are MSG recommendations (§3.2–3.3 and §3.1.4). Use text lists since you cannot create PDF bookmarks.

## 3) Core annotation rules (adapted from MSG)

### Scope & searchability
- Annotate only SDTM tabulation data intended for submission; exclude internal operational/system fields. (General note in §3.1).
- If a variable/page was planned but no data were collected, still annotate as if collected, and set collected_boolean=true; the absence is conveyed in Define‑XML via HasNoData. Add a notes hint like "Planned; no data collected in study" (§3.1).

### Domain vs dataset
- Put the domain in domain (e.g., "QS") and—if applicable—put the split physical dataset in dataset_name (e.g., "QSPH", "QSSL"). (MSG shows QS split by sponsor choice; LB split by category; EC/EX linkage) (§5.5.1, §5.5.3, §5.3.1).

### Variables & formatting
- Variables and dataset codes are ALL CAPS; when annotating multiple variables together, separate with "/" (e.g., VSORRES/VSORRESU). Use sentence case for free‑text notes. (§3.1.2).

### Findings domains need TESTCD context
- For vertical Findings (e.g., VS, LB, QS/FT/RS), add a when clause:
  "<VAR(S)> when <TESTCD>=<VALUE>" (e.g., VSORRES/VSORRESU when VSTESTCD=TEMP). (§3.1.3).

### Units, repeats, timepoints
- Capture units as --ORRESU when present; repeated measures use --REPNUM (e.g., VSREPNUM, FTREPNUM). Use repeat_number_var if repeats are implied by text like "Triplicate". (§3.1.1 example and §5.5.4; FT example in §5.5.3).

### Origins
- Populate origin with one of: "Collected", "Derived", "Predecessor" (e.g., many EX vars copied from EC are "Predecessor"), "Assigned", or "ePRO Collected". (§5.3.1; origin values and notes in §3.1.1).

### Supplemental qualifiers
- If the CRF has an "Other, specify" or similar free text not mapped to a standard variable, annotate the QNAM and SUPPxx: e.g., "qnam":"RACEOTH","supp_domain":"SUPPDM". (§3.1.3 "Supplemental Qualifiers").

### RELREC relationships
- If a form explicitly relates records (e.g., death details capturing an AE ID), add:
  "relrec": {"has_relrec": true, "statement": "RELREC when DDLNKID = AE.AELNKID"}. (§3.1.3 "RELREC").

### Not‑submitted prompts
- If the CRF contains prompts collected operationally but not submitted to SDTM, set "not_submitted": true and include [NOT SUBMITTED] in notes. (§3.1.2 #12).

### Unique pages & versioning
- Prefer unique forms; if versions exist (e.g., VS v1.0 vs v2.0), keep both in output and explain differences in notes. (§3.1.1 & §3.1.4).

### Bookmarks/TOC (text simulation)
- Return document_toc.by_visits (chronological, with "Running Records" last) and by_forms (alphabetical or CRF order). (§3.2–3.3).

### Codelists
- Where the CRF shows enumerations, set a short codelist_hint (e.g., "RACE choices: …"). (Define‑XML/codelist guidance in §2).

## 4) Examples

### A. Vital Signs – Temperature
CRF text: "Temperature: ____ (°C)"

```json
{
  "form_id": "Vital Signs",
  "page_version": "v2.0",
  "visit_context": "Week 4",
  "text_anchor": "Temperature (°C)",
  "domain": "VS",
  "dataset_name": null,
  "variables": ["VSORRES","VSORRESU"],
  "when": "VSORRES/VSORRESU when VSTESTCD=TEMP",
  "origin": "Collected",
  "collected_boolean": true,
  "not_submitted": false,
  "supplemental": {"is_supplemental": false, "qnam": null, "supp_domain": null},
  "relrec": {"has_relrec": false, "statement": null},
  "repeat_number_var": "VSREPNUM",
  "units_var": "VSORRESU",
  "codelist_hint": null,
  "notes": "Triplicate BP/pulse on this version; repeats identified by VSREPNUM."
}
```
(MSG: Findings "when" pattern; repeats via VSREPNUM; v2.0 triplicate example in §3.1.3 and §5.5.4.)

### B. Demographics – Race (Other, specify)
CRF text: "Race: [White][Black][Asian][Other: ____]"

```json
{
  "form_id": "Demographics",
  "page_version": null,
  "visit_context": "Screening",
  "text_anchor": "Race (Other: ____)",
  "domain": "DM",
  "dataset_name": null,
  "variables": ["RACE"],
  "when": null,
  "origin": "Collected",
  "collected_boolean": true,
  "not_submitted": false,
  "supplemental": {"is_supplemental": true, "qnam": "RACEOTH", "supp_domain": "SUPPDM"},
  "relrec": {"has_relrec": false, "statement": null},
  "repeat_number_var": null,
  "units_var": null,
  "codelist_hint": "RACE choices include 'Other'",
  "notes": "Free text in 'Other, specify' goes to SUPPDM as QNAM=RACEOTH."
}
```
(MSG: Supplemental qualifier annotation uses QNAM + SUPPxx; see §3.1.3 "Supplemental Qualifiers".)

### C. Death Details – AE linkage
CRF text: "Related AE ID: ______"

```json
{
  "form_id": "Death Details",
  "page_version": null,
  "visit_context": "Running Record",
  "text_anchor": "Related AE ID",
  "domain": "DD",
  "dataset_name": null,
  "variables": ["DDLNKID"],
  "when": null,
  "origin": "Collected",
  "collected_boolean": true,
  "not_submitted": false,
  "supplemental": {"is_supplemental": false, "qnam": null, "supp_domain": null},
  "relrec": {"has_relrec": true, "statement": "RELREC when DDLNKID = AE.AELNKID"},
  "repeat_number_var": null,
  "units_var": null,
  "codelist_hint": null,
  "notes": "Establish explicit relationship to AE domain."
}
```
(MSG: RELREC pattern "RELREC when <collected var> = <related domain var>"; §3.1.3 "RELREC".)

### D. Questionnaire split (QS → QSPH)
CRF text: "PHQ‑9 Question 1 … (0–3)"

```json
{
  "form_id": "Patient Health Questionnaire-9 (PHQ-9)",
  "page_version": null,
  "visit_context": "Week 2",
  "text_anchor": "PHQ-9 Q1 (0–3)",
  "domain": "QS",
  "dataset_name": "QSPH",
  "variables": ["QSTESTCD","QSORRES","QSSTRESC","QSSTRESN"],
  "when": "QSORRES/QSSTRESC/QSSTRESN when QSTESTCD=PHQ1",
  "origin": "Collected",
  "collected_boolean": true,
  "not_submitted": false,
  "supplemental": {"is_supplemental": false, "qnam": null, "supp_domain": null},
  "relrec": {"has_relrec": false, "statement": null},
  "repeat_number_var": null,
  "units_var": null,
  "codelist_hint": "0–3 Likert scale",
  "notes": "QS split by instrument; standardized results included for clarity though derived."
}
```
(MSG: QS split example; derived standardized results can be annotated for clarity but remain "Derived"; §5.5.3.)

### E. Operational prompt not submitted
CRF text: "If 'Yes', explain reason for late entry: ______ (for site use only)"

```json
{
  "form_id": "ConMeds",
  "page_version": null,
  "visit_context": "Week 8",
  "text_anchor": "reason for late entry (for site use only)",
  "domain": "CM",
  "dataset_name": null,
  "variables": [],
  "when": null,
  "origin": "Operational",
  "collected_boolean": false,
  "not_submitted": true,
  "supplemental": {"is_supplemental": false, "qnam": null, "supp_domain": null},
  "relrec": {"has_relrec": false, "statement": null},
  "repeat_number_var": null,
  "units_var": null,
  "codelist_hint": null,
  "notes": "[NOT SUBMITTED] Operational field; exclude from SDTM."
}
```
(MSG: mark collected‑but‑not‑submitted prompts as "[NOT SUBMITTED]"; §3.1.2 #12.)

## 5) Validation checks the model must perform
- Every Findings variable that can vary by test (--ORRES, --ORRESU, --STRESC/N) has a when clause referencing --TESTCD. (§3.1.3).
- Supplemental data uses supplemental.is_supplemental=true, supplies qnam and supp_domain. (§3.1.3).
- RELREC present when an explicit cross‑form link is captured on the CRF. (§3.1.3).
- Origins are populated and consistent with MSG examples (e.g., EC→EX "Predecessor"). (§5.3.1).
- Versioning: if text indicates "Version x.y", ensure page_version and replaced_or_deprecated_pages are filled. (§3.1.4).
- Bookmarks: supply document_toc.by_visits (SoA order; "Running Records" last) and by_forms. (§3.2–3.3).

## 6) Style rules the model must follow (text replacements for visual cues)
- Use ALL CAPS for domain/dataset/variable codes. Notes are sentence case.
- Combine variables with "/" in a single annotation when they belong to the same field (e.g., result and unit).
- Use [NOT SUBMITTED] inside notes to stand in for the dashed border convention for non‑submitted items. (§3.1.2).

## 7) What not to do
- Don't annotate sponsor internal operational fields unless needed for a RELREC or explanatory note. (§3.1 General Note).
- Don't omit --TESTCD context for Findings. (§3.1.3).
- Don't rename domains to split dataset names; keep domain (e.g., "QS") and put the split in dataset_name (e.g., "QSPH"). (§5.5.1, §5.5.3).