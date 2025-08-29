"""Extraction prompts for different CRF elements"""

QUESTION_EXTRACTION_PROMPT = """Extract ALL questions and their input fields from this CRF page.
        YOU MUST SCAN THE PAGE FROM TOP TO BOTTOM CAREFULLY. Do not miss any detail 
Hint: questions usually share same format and font size, if something looks different than other questions in format, font size or colour, it is not a question

OUTPUT FORMAT:
- For each question, output: <Q> followed by the question text
- For each input field, output: <INPUT> followed by the input text
- Input fields should immediately follow their question
- One item per line
- You must only output questions

EXAMPLE:
<Q> Date of visit
<INPUT> dd-mmm-yyyy
<Q> Is the subject eligible?
<INPUT> Yes
<INPUT> No
<Q> Subject ID
<INPUT> _______________

Extract everything - questions, checkboxes, text fields, date fields, etc.
Output ONLY the tagged lines, nothing else:"""

SECTION_EXTRACTION_PROMPT = """Extract ONLY section headers from this CRF page and identify the first question below each section.

A section header is:
- A title for a group of related questions
- Usually in bold, larger font, or different color
- Examples: Demographics, Medical History, Vital Signs, Laboratory Tests

NOT a section header:
- Questions (anything ending with ?)
- Input fields or options
- The form name: {form_name}
- These already found questions:
{questions_list}

YOU MUST NOT OUTPUT THE FOUND QUESTIONS AND FOUND FORM NAME
YOU MUST TO LOOK ALL OTHER CONTENT IN PAGE, FIND ALL POSSIBLE SECTION HEADER, BUT IT'S OK NOT FIND ONE
YOU MUST ONLY PARSE WHOLE TEXT, DO NOT PARSE PART OF IT
YOU MUST SCAN THE PAGE FROM TOP TO BOTTOM CAREFULLY. Do not miss any detail 

For each section header, identify the FIRST question that appears below it on the page.

OUTPUT FORMAT:
- For each section header, output: <SH> followed by the header text, then a pipe |, then the first question below it (or "none" if no question below)
- If no section headers found, output: SH_NOT_FOUND
- One item per line

EXAMPLE:
<SH> Demographics | Date of birth (DOB)
<SH> Inclusion Criteria | Age ≥ 18 years (INC1)
<SH> Physical Examination | none

Output ONLY the tagged lines or SH_NOT_FOUND:"""

FORM_NAME_EXTRACTION_PROMPT = """Find the main FORM NAME on this CRF page.

The form name is the overall title of the entire form.
Examples: Baseline Visit, Screening Visit, Follow-up Visit Week 12, Enrollment Form

Look for:
- Title at the top of the page
- Text in header/footer
- Largest title on the page

OUTPUT FORMAT:
- Output: <FORM> followed by the form name
- If no form name found, output: FORM_NOT_FOUND
- Only ONE form name per page

EXAMPLE:
<FORM> Baseline Visit

Output ONLY the tagged line or FORM_NOT_FOUND:"""