# COSMoS knowledge source

The local snapshot adds **1,475 biomedical concepts**, **1,475 SDTM dataset
specializations**, and **303 draft CRF specializations** to the project's available
knowledge. These are distinct entities grouped from 6,306, 13,922, and 2,073 CSV
data records respectively. Repeated variables remain scoped to their template.

Source: [CDISC COSMoS](https://github.com/cdisc-org/COSMoS), fixed commit
`031429b1d14823721991cd23ee88a11616686ce3`.

[Snapshot manifest](031429b1d14823721991cd23ee88a11616686ce3/manifest.json)
records the exact source URLs, SHA-256 hashes, entity counts and default filters.
The original exports and upstream license notices are retained in the snapshot's
`raw/` directory. No CDISC API key, network request, embedding model or mapping
model is needed to load or query this source.

## Use from the repository root

```sh
python3 -m sdtm_knowledge_base.cosmos verify
python3 -m sdtm_knowledge_base.cosmos show sdtm WEIGHT
python3 -m sdtm_knowledge_base.cosmos show bc C81328
python3 -m sdtm_knowledge_base.cosmos show crf WEIGHT_DENORMALIZED --include-drafts
python3 -m sdtm_knowledge_base.cosmos evidence --kind sdtm --domain VS
```

`show` is exact concept/template lookup, not natural-language mapping. `evidence`
exports complete records with `id`, `kind`, `text`, source provenance and structured
rows. Keeping the whole specialization lets a flat RAG baseline see the same
conditions and variable relationships that a graph-based consumer would use.
The CLI emits JSON to stdout and returns exit code 2 for invalid inputs.

```python
from sdtm_knowledge_base.cosmos import load_snapshot

catalog = load_snapshot()
weight = catalog.lookup("sdtm", "WEIGHT").to_dict()
documents = catalog.evidence_documents(kind="sdtm", sdtmig_version="3.4")
entity_links = catalog.links(sdtmig_version="3.4")
```

`KBLoader.load_cosmos_resources()` exposes the same catalog through the existing
Stage B KB loader. Calling `load_kb_resources(include_cosmos=True)` loads it into
`loader.cosmos` alongside existing resources. The default is opt-out so that
current mapper runs remain the original comparison baseline. Adding the source
does not automatically change the existing domain/variable/CT ranking or insert
COSMoS context into model prompts. Consumers explicitly request its evidence.

## Interpretation and version boundaries

- Biomedical concepts and SDTM specializations come from the upstream published
  `*_latest.csv` exports at the pinned commit, not a sum of historical YAML packages.
- CRF templates come from `cdisc_crf_specializations_draft.csv`. They keep
  `status: draft` and require `include_drafts=True` / `--include-drafts`.
- SDTM lookups default to SDTMIG 3.4. Source start/end versions are inclusive,
  matching the [upstream model](https://github.com/cdisc-org/COSMoS/blob/031429b1d14823721991cd23ee88a11616686ce3/model/cosmos_sdtm_model.yaml).
  `sdtmig_version=None` in Python explicitly disables this filter for inspection.
- The SDTMIG filter applies to SDTM specializations. CRF records preserve their
  separate CDASHIG version metadata; including a draft does not certify its
  compatibility with a particular SDTMIG version.
- Dates and all original CSV cell values remain intact, including semicolon lists,
  assigned values, origins, conditions and subject/predicate/object relationships.
  A row's `record_number` is its 1-based CSV data-record number, excluding the
  header; quoted multiline cells mean this is not a physical text line number.
- `links()` resolves explicit entity IDs only and returns edges whose endpoints
  pass the same draft/version filters. Variable relationships remain within their
  complete specialization rows. The importer does not infer relationships from prose.
- The snapshot reports **70 unresolved references**: 68 variable references outside
  their specialization's variable list and two CRF-to-SDTM references (`EQ5D02`,
  `SIXMW10`) absent from the selected SDTM export. Some are intentional cross-domain
  references; they are not all standard errors. See [issues.json](031429b1d14823721991cd23ee88a11616686ce3/issues.json).
- A verified snapshot proves source-byte fidelity, not clinical correctness.
  `CosmosCatalog.from_exports()` also accepts caller-provided CSV bytes for testing;
  those records explicitly have `verified_snapshot: false` and no claimed upstream
  commit, URL or license.

## Reproduce and validate

The shipped raw files can rebuild a second snapshot without any download:

```sh
python3 -m sdtm_knowledge_base.cosmos import \
  --source-directory kb/cosmos/031429b1d14823721991cd23ee88a11616686ce3/raw \
  --output-directory /tmp/cosmos-rebuilt
python3 -m sdtm_knowledge_base.cosmos verify --snapshot /tmp/cosmos-rebuilt
python3 -m unittest test_cosmos_kb -v
```

Imports verify all five pinned source files before writing, stage a complete
snapshot, and refuse to overwrite an existing destination. Loading verifies the
source hashes and reproduces the manifest, notices and reference report. Updating
the source revision is an explicit code/data change, not an automatic `main` sync.

The tests exercise literal fixtures, the complete real snapshot, evidence
serialization, the existing KB loader and the CLI. They assert full weight mappings,
cross-template isolation, version boundaries, draft exclusion, malformed input,
unresolved references, tamper rejection and byte-identical rebuilds. No mapping
accuracy or GraphRAG improvement is claimed by these checks.

## Attribution

Content based on [COSMoS (GitHub)](https://github.com/cdisc-org/COSMoS), used under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Raw CSV files are unchanged;
local grouping, evidence serialization, manifests and reports are aCRF additions.
Upstream code/scripts have a separate MIT license. Preserve this attribution and
identify modifications when redistributing derived content; trademark permissions
are not included.
