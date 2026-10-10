# Run the standards annotation pipeline

[Project overview](../README.md) · [Architecture](architecture.md) · [Benchmark](benchmark.md)

## Python API

The new standards-only core is exposed through Python. The legacy
`process-file` command still uses the older mapper; it does not select this
workflow. A new unified command-line entry point is not part of this version.

Run the following from the repository root. Python 3.12 is the tested runtime.
The annotation client uses Pydantic and Python's standard library; the model
server runs separately.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "pydantic==2.13.4"
```

Two prerequisites are required for inference:

1. A verified standards corpus at
   `kb/standard_mapping_cdash23_sdtmig34_20261004/`. It contains `manifest.json`
   plus the pinned CDASH IG, CDASH Model, SDTM IG, SDTM Model, terminology, and
   COSMoS collection/specialization exports. `StandardsCorpus` verifies their
   profile and byte hashes when loading them. The current local corpus is not
   an automatic download or installation step; its distribution must follow
   the source materials' terms.
2. A running local model server at `http://127.0.0.1:1234`, with a loaded model
   supporting the complete request context and native JSON Schema output.
   Set `model` below to its actual loaded identifier. The measured model used
   a 208,384-token context configuration. Other models or servers need separate
   validation and do not inherit the reported results.

The current client accepts loopback HTTP endpoints only. The call writes a new
trace directory and refuses to overwrite an existing one.

```sh
python - <<'PY'
import json
from pathlib import Path

from pipeline.stage_b_mapper_module.annotation_workflow.runtime import LocalClient
from pipeline.stage_b_mapper_module.annotation_workflow.standard_annotation import run_standard
from pipeline.stage_b_mapper_module.annotation_workflow.standard_context import StandardsCorpus

case = json.loads(Path("examples/standards_annotation/input.json").read_text())
corpus = StandardsCorpus(Path("kb/standard_mapping_cdash23_sdtmig34_20261004"))
client = LocalClient(
    # Local serving alias; replace with your server's loaded model identifier.
    model="acrf-qwen38-27b-4bit",
    base_url="http://127.0.0.1:1234",
    schema_transport="native",
    reasoning_effort="none",
)
output = Path("annotations/evaluator-example")
result = run_standard(case, output, client, corpus)
print(json.dumps(result["outcome"], indent=2))
print((output / "standard-answer.json").read_text())
PY
```

Inspect both `standard-answer.json` and `outcome.json`. An `incomplete` outcome,
`abstain`, or a partial decision needs investigation or additional context.
Starting the client does not load the model. Use a fresh output directory for
each run. The workflow allows up to three correction attempts within its
resource budget and records the requests and replies locally.

## Checks without model inference

The agent's search, evidence boundaries, question tracking, and recovery checks
use the included COSMoS snapshot and scripted model replies:

```sh
PYTHONPATH=tests/annotation_workflow python -m unittest \
  test_action_workflow test_evidence_actions \
  test_question_commit test_recovery_loop -q
```

With the verified corpus in place, these focused tests exercise public-context
retrieval, mapping requirements, validation, semantic export, and correction
behavior. Inference is replaced by controlled fixtures; no live model is called.

```sh
python -m unittest \
  tests.annotation_workflow.test_standard_context \
  tests.annotation_workflow.test_standard_requirements \
  tests.annotation_workflow.test_standard_run \
  tests.annotation_workflow.test_standard_wire -q
```
