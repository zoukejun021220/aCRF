# aCRF: evidence-grounded clinical form annotation

aCRF is a research project for mapping blank clinical Case Report Form (CRF)
questions to Study Data Tabulation Model (SDTM) annotations with local language
models. The task includes interpreting context, conditions, and record identity
alongside variable selection.

The project implements **model-directed evidence search, structured annotation
contracts, validation with bounded correction, and reproducible evaluation**.
A standards-only pipeline reached **96.07% complete semantic agreement on 1,196
scoreable public-template cases**, out of 1,211 attempted, against an
assistant-compiled reference set.

[Architecture](docs/architecture.md) · [Example](#example) ·
[Results and benchmark](docs/benchmark.md) · [Run locally](docs/usage.md)

## Architecture

![Agentic annotation workflow: source interpretation and initial retrieval lead to model-selected evidence requests or a mapping proposal. Search results and validation feedback return to the decision step; accepted proposals produce reviewable SDTM mappings.](docs/figures/agentic-architecture.svg)

| Workflow | What it does | Evidence available |
| --- | --- | --- |
| **Agentic workflow** | Interprets a source packet, retrieves initial candidates, then lets the model choose `search_standard`, `read_material`, or `submit` | Implemented controller and integration fixtures; its search policy is not evaluated by the 96.07% result |
| **Standards-only pipeline** | Retrieves pinned public context once, then generates, validates, and corrects a structured annotation | Full 1,211-case evaluation, selected input/output, sealed report, and all remaining discrepancies |

## Engineering decisions

- **Scoped evidence actions.** Search and read actions identify the
  question, affected source fields, and intended impact. The controller checks
  scope and material handles before executing them.
- **Annotation contracts.** Source bindings, constants, conditions, and target
  records represent conditional branches and distinguish repeated measurements.
- **Controlled revision.** The agent tracks open, resolved, and deferred
  questions. Revisions preserve unaffected accepted work, and call/token/time
  budgets plus no-progress checks bound execution.
- **Reproducible evaluation.** Pinned sources, saved request/reply bytes,
  separate offline references, and complete residual reporting make the measured
  result traceable. Fixture tests exercise failure handling independently of
  live-model quality.

## Example

Standards-only annotation of a field in a blank collection form:

| Input | Value |
| --- | --- |
| Focused field | `F012`: “What is the evaluator identifier?” |
| Form context | Disease Response and Clin Classification |
| Collection instruction | Identify the evaluator providing this evaluation. |
| Related field | `F015` collects the disease response / clinical classification test name |

The output binds the focused field to `RS.RSEVALID` and uses the neighboring
test-name field to identify the target record:

```json
{
  "status": "mapped",
  "facts": [
    {
      "domain": "RS",
      "variable": "RSEVALID",
      "value": {"kind": "source", "field_id": "F012"},
      "when": null,
      "record": {"kind": "collection_record", "test_source_field": "F015"}
    }
  ],
  "missing": [],
  "not_submitted_when": [],
  "explanation": "CDASH 1.3 RSEVALID maps directly to SDTMIG 3.4 RS.RSEVALID; the collected test name F015 binds the collection record."
}
```

[Input JSON](examples/standards_annotation/input.json) ·
[Output JSON](examples/standards_annotation/output.json)

## Measured results

The standards-only run used **Qwen3.8-27B, MLX 4-bit, through LM Studio**, with
native JSON Schema output. **No aCRF-specific fine-tuning or LoRA adapter was
used.** The `acrf-qwen38-27b-4bit` name in request logs is a local serving alias.
[Inference configuration](docs/benchmark.md#model-and-inference-configuration)

| Measure | Matching cases | Agreement |
| --- | ---: | ---: |
| Complete semantic decisions | 1,149 / 1,196 | **96.07%** |
| Standards-resolvable decisions, including supported `NOT SUBMITTED` | 991 / 995 | 99.60% |
| Appropriate uncertainty decisions | 158 / 201 | 78.61% |

All 1,211 cases were attempted. Fifteen have conflicting or pending references
and remain unscored. Of the 47 scored discrepancies, **43 concern uncertainty**;
recognizing missing conventions remains the main source of disagreement.

The benchmark comes from **CDASHIG 2.3 / CDASH Model 1.3** collection metadata
and a pinned **CDISC COSMoS** snapshot. Its 1,284 inventoried rows produce 1,211
prediction cases across 44 domain or instrument families. References use
published mapping instructions, **SDTMIG 3.4 / SDTM 2.0**, and pinned terminology.
Published collection metadata and mapping instructions support reproducible
comparisons across domains, including cases that require additional study
conventions. [Benchmark construction and provenance](docs/benchmark.md)

These results assess the combined retrieval, model, and validation pipeline on
metadata-derived templates. References have not received independent expert
review. Agentic-search performance, accuracy on unseen study forms, and
end-to-end PDF annotation quality remain unevaluated.

[Verified report](docs/results/standard-annotation-v1-report.json) ·
[All 47 discrepancies](docs/results/standard-annotation-v1-residuals.json) ·
[Partition results and limitations](docs/benchmark.md#results)

## Usage

[Installation, inference, and tests](docs/usage.md)

Full inference requires a separately prepared standards corpus. A clean-clone
corpus setup remains in preparation. The current standards-only entry point is
`run_standard`; the older `process-file` CLI still selects the legacy mapper.

## Code structure

| Area | Entry point | Responsibility |
| --- | --- | --- |
| Agent actions and state | [session.py](pipeline/stage_b_mapper_module/annotation_workflow/session.py), [evidence_actions.py](pipeline/stage_b_mapper_module/annotation_workflow/evidence_actions.py) | Model-selected tools, question tracking, targeted revision, and stopping |
| Agent retrieval | [knowledge.py](pipeline/stage_b_mapper_module/annotation_workflow/knowledge.py) | SQLite full-text search with concept, property, and context ranking |
| Measured annotation pipeline | [standard_annotation.py](pipeline/stage_b_mapper_module/annotation_workflow/standard_annotation.py), [standard_context.py](pipeline/stage_b_mapper_module/annotation_workflow/standard_context.py) | Public-context retrieval, typed proposals, validation, and bounded correction |
| Output and state integrity | [standard_wire.py](pipeline/stage_b_mapper_module/annotation_workflow/standard_wire.py), [store.py](pipeline/stage_b_mapper_module/annotation_workflow/store.py) | Wire-to-semantic conversion, atomic artifacts, and decision snapshots |
| Behavioral checks | [test_action_workflow.py](tests/annotation_workflow/test_action_workflow.py), [test_question_commit.py](tests/annotation_workflow/test_question_commit.py) | Real search with scripted model replies, invalid evidence rejection, and state publication |

Shared annotation contracts and source verification live in
`pipeline/stage_b_mapper_module/graphrag/`. The PDF digitizer in
`pipeline/stage_a_digitizer_module/` and the training packages under
`training/` and `sdtm_instruction_tuning_package/` belong to other or historical
workflows. They are not prerequisites for the measured path.

Explicit terminology-provenance output and final PDF annotation placement remain
outside the measured version.
