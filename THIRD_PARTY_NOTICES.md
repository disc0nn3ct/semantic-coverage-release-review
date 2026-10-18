# Third-party materials

The root MIT license covers original research scripts and documentation authored
for this artifact. It does not relicense third-party source code, source records,
papers, tokenizers, or copied benchmark material.

- FastAPI, Apache Airflow, Home Assistant Core and Keycloak: preserve the original
  commit/PR provenance and license snapshots in LICENSES/upstream/ and the data
  archive. Airflow NOTICE is retained. A missing upstream Keycloak NOTICE fetch
  remains recorded as unavailable, rather than claimed to exist.
- CASCADE and DocPrism benchmark repositories provide MIT notices. Copies are in
  LICENSES/CASCADE-MIT.txt and LICENSES/DocPrism-MIT.txt. Their paper copies declare
  CC BY 4.0; attribution remains in the archived paper records.
- Some original Java source excerpts in the CASCADE dataset retain GNU GPL
  version 2 or later notices. These original terms apply to those excerpts,
  notwithstanding the benchmark repository's top-level license. The GPLv2 text
  accompanies this distribution in LICENSES/GPL-2.0.txt. In the CASCADE source
  tree the affected paths are:
  `java/graphhopper/jsprit/{05dc00814e3f71a8661e2586594b842d583f533c,fa8d97b7e9e340cde8b61c84d8f5bafd8d2580bc}/1/analyzed.json`.
  They occur in PaperEvaluation/coreDataset.zip, PaperEvaluation/dataset.zip and
  selected_core/ copies, under external_gold_validation_20260930/sources/CASCADE/.
  The original source headers and source contents are preserved. No binary
  build of these projects is distributed here.
- Embedding tokenizer/model-card files retain their recorded upstream identities
  and applicable notices. Generator and encoder model weights are not distributed.

Journal template/class/logo files and the manuscript-source bundle are excluded
from this code/data release. The article and its supplement are separate
submission documents. No license to the journal's template assets is asserted.
