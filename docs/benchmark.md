# Standards annotation benchmark

The reported standards workflow achieved **96.07% complete semantic agreement
(1,149 / 1,196 scoreable cases)** after attempting all 1,211 benchmark cases.
This document records the sources, construction, model configuration, results,
and interpretation limits behind that result.

The benchmark is constructed from public CDISC collection metadata and pinned
mapping instructions. Its references are assistant-compiled and have not been
independently adjudicated by experts. It is a project benchmark, with no CDISC
endorsement.

[Return to the project overview](../README.md).

## What was measured

The measured path is
[`standard_annotation.run_standard`](../pipeline/stage_b_mapper_module/annotation_workflow/standard_annotation.py).
For one focused blank CRF field and its neighbors, it:

1. Retrieves applicable public standards context through
   [`StandardsCorpus.retrieve`](../pipeline/stage_b_mapper_module/annotation_workflow/standard_context.py).
   This is deterministic retrieval based on form, section, and collection
   wording, completed before the first model request.
2. Requests a structured annotation from the local model.
3. Validates the proposed source bindings, target facts, conditions, record
   identity, and missing conventions; returns bounded correction feedback when
   needed.
4. Exports reviewable annotation JSON, or `abstain` if no proposal passes
   validation. The exporter does not invent replacement facts.

The run permits at most three model attempts within an 8-call, 60,000-token,
900-second resource ceiling per case. Requests, replies, validation feedback,
retrieved context, and outcomes are retained in the run trace.

These results apply to that combined retrieval, model, and validation path.
The separate [`Session` agentic workflow](../pipeline/stage_b_mapper_module/annotation_workflow/session.py)
allows model-selected evidence actions. Its search behavior, and the effect of
knowledge-graph traversal, are not measured by the result reported here.

## Sources and pinned versions

| Resource | Pinned version or snapshot |
| --- | --- |
| CDASH Implementation Guide / Model | 2.3 / 1.3 |
| SDTM Implementation Guide / Model | 3.4 / 2.0 |
| SDTM controlled terminology | 2025-03-28 |
| CDISC CORE metadata caches | `f26422c2258ace774d0517ec11444b5413118882` |
| COSMoS exports | `031429b1d14823721991cd23ee88a11616686ce3` |

| Source | Contribution |
| --- | --- |
| [CDASHIG 2.3](https://www.cdisc.org/standards/foundational/cdash/cdashig-v2-3) and CDASH Model 1.3 | 1,253 inventoried collection fields: 779 domain-table fields and 474 scenario fields; question text, collection instructions, and form context |
| [CDISC COSMoS exports](https://github.com/cdisc-org/COSMoS) | 31 draft digital collection rows for EQ-5D-5L and Six Minute Walk, including eight hidden fields retained as context |
| [SDTMIG 3.4](https://www.cdisc.org/standards/foundational/sdtmig/sdtmig-v3-4), SDTM Model 2.0, and SDTM terminology | Target definitions, model inheritance, and terminology used to interpret and check mapping references |

CDASH and SDTM metadata were exported from the official
[CDISC CORE repository's caches](https://github.com/cdisc-org/cdisc-rules-engine)
at the pinned commit above. A source lock records hashes of the original files
and their JSON exports, so later upstream changes do not silently change the
benchmark. The source-lock SHA-256 is:

```text
f8d369158219858330cc4a6a32927886a293dff85f58d445f0f6cb34d2e56505
```

## How the cases and references were constructed

1. **Inventory the source population.** Start with 1,284 source rows. Exclude
   65 assigned or non-collected fields and eight hidden QRS context fields from
   prediction tasks. This leaves **1,211 cases across 44 domain or instrument
   families**: 715 domain-table cases, 473 scenario cases, and 23 visible QRS
   cases.
2. **Construct blank collection inputs.** Each focused field includes its
   question, collection instructions, declared datatype, and neighboring fields.
   Input metadata excludes target-variable identifiers and mapping-note
   properties. There are no patient responses, sponsor study specifications,
   or PDF coordinates. Declared datatypes come from metadata rather than
   observations of PDF widgets.
3. **Compile reference decisions.** Published mapping instructions and target
   links provide conditions, constants, transformations, and record bindings.
   Supported alternatives and missing conventions are preserved. Draft QRS
   annotations are not copied directly as references: scored item identities
   are corroborated against approved dataset specializations and the pinned
   terminology release.
4. **Separate inference from scoring.** Public standards retrieval is allowed
   during inference; accepted evaluator answers are not supplied to the model.
   Sealed model outputs are scored offline. Of the 1,211 cases, 1,196 have
   scoreable references; 11 source conflicts and four pending references remain
   explicitly unscored.

The scoreable reference set contains 969 mapped cases, 26 supported
`NOT SUBMITTED` decisions, and 201 partial or `needs_convention` decisions.
Missing evidence alone does not justify `NOT SUBMITTED`.

Whole domain or instrument families are separated between development and
holdout. Shared identifier and visit boilerplate is assigned to calibration.
The family holdout tests a public-knowledge partition; it does not establish
generalization to prospectively collected study forms.

### Why this benchmark is useful

It tests whether a pipeline follows published collection-to-SDTM mappings
consistently. The complete inventoried population is attempted, source versions
are fixed, and scoring checks full annotation semantics: conditions,
transformations, target-record bindings, and missing conventions in addition to
variable names. Explicit uncertainty cases test whether the pipeline recognizes
when the available standards do not resolve a decision.

This makes the benchmark useful for controlled pipeline comparisons and
regression analysis within the stated source population. The limitations below
define how far those results can be generalized.

## Model and inference configuration

**No aCRF-specific fine-tuning or LoRA adapter was used for this reported run.**
It used the published Qwen3.8-27B model through the community
[Qwen3.8-27B-MLX-4bit checkpoint](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit),
served locally by LM Studio on an Apple GPU. The model publisher describes this
checkpoint as an MLX 4-bit quantization of the original Qwen model. Quantization
reduces weight precision; it is not training on CRF or SDTM examples.

| Identifier | Meaning |
| --- | --- |
| `Qwen/Qwen3.8-27B` | Upstream model repository |
| `lmstudio-community/Qwen3.8-27B-MLX-4bit` | Published quantized checkpoint installed locally |
| `qwen/qwen3.8-27b` | LM Studio model key recorded by the run |
| `acrf-qwen38-27b-4bit` | Local serving alias used in requests; it does not identify a separately trained aCRF model |

Requests used native JSON Schema output, temperature 0, and reasoning disabled.
The loaded model used a 208,384-token context configuration. The annotation
client ran under Python 3.12 and connected to a local loopback HTTP server.
Other models or server configurations require separate validation.

The project-specific components are standards retrieval, context construction,
prompts, structured output, and validation. The benchmark evaluates these
components together with the model, rather than measuring the language model
in isolation.

## Results

Every benchmark case was attempted with fresh inference. No earlier predictions
were reused.

| Measure | Matching cases | Rate |
| --- | ---: | ---: |
| Complete semantic decisions, including uncertainty | 1,149 / 1,196 | 96.07% |
| Standards-resolvable decisions, including supported `NOT SUBMITTED` | 991 / 995 | 99.60% |
| Appropriate uncertainty decisions | 158 / 201 | 78.61% |

Fifteen attempted cases remain unscored: 11 source conflicts and four pending
references. Forty-seven scored cases differ from the accepted references:
four resolvable mappings and 43 uncertainty decisions. The 99.60% figure is
specific to the standards-resolvable subset and must not be presented as
accuracy over all inputs.

| Partition | Attempted | Complete semantic agreement | Standards-resolvable agreement |
| --- | ---: | ---: | ---: |
| Development | 738 | 709 / 724 (97.93%) | 622 / 623 (99.84%) |
| Holdout | 227 | 211 / 226 (93.36%) | 162 / 165 (98.18%) |
| Calibration | 246 | 229 / 246 (93.09%) | 207 / 207 (100%) |

The run made **1,367 model calls**, with **13,005,672 measured tokens**. Total
invocation time was about **10 hours 43 minutes** (38,606.166 seconds). Timing
is a single-machine observation.

Verification checked all 1,211 case seals, the 1,367 actual wire requests, model
settings, archived dependencies, and sealed scores. These checks establish
consistency of the recorded run artifacts; they do not independently validate
the reference interpretations.

## Inspect the evidence

- [Full partition report and verification metadata](results/standard-annotation-v1-report.json)
- [All 47 residual discrepancies](results/standard-annotation-v1-residuals.json)
- [One complete blank input](../examples/standards_annotation/input.json)
  and its [measured output](../examples/standards_annotation/output.json)

The report and residual files preserve the original sealed run's contents.
The example output was exported by the measured local pipeline; viewing it does
not run new inference. These selected artifacts can be read without downloading
the bulk experiment archive.

The benchmark manifest SHA-256 is:

```text
fd8789e6b5c7b2f7351886e9b68f64f3274a2b98953753541ea0574181cede0c
```

## Interpretation and remaining limits

- **Reference independence:** references were produced by assistant-authored
  compilation rules, without independent expert adjudication. Public mapping
  instructions also inform the pipeline's validator, so shared interpretation
  errors can affect both the pipeline and its references.
- **Generalization:** the inputs represent public collection templates. The
  benchmark does not establish performance on custom study wording,
  study-specific conventions, actual PDF extraction, repeated PDF layouts,
  or annotation placement.
- **Annotation scope:** outputs describe annotation operations and do not
  populate patient datasets. Passing local constraints does not establish full
  clinical submission conformance. Explicit codelist and term-code provenance
  output is deferred in this version.
- **Reproducibility and distribution:** full inference requires the verified
  standards corpus. A clean-clone corpus setup and public distribution package
  remain in preparation; standard definitions and exports retain their source
  terms.
- **Comparability:** historical sponsor-agreement benchmarks and alternate-model
  experiments measure different tasks or configurations. They need their own
  labels and denominators. This report does not evaluate the separate agentic
  `Session` workflow or establish a benefit from graph traversal.

The residual set shows that most remaining disagreement concerns uncertainty:
43 of the 47 discrepant decisions are from that category. Independent reference
review and better handling of missing conventions are concrete next steps;
unseen-study evaluation is needed before making broader accuracy claims.
