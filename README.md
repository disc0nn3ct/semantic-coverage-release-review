# Semantic Coverage for Release Review

Code and saved-output replication artifacts for a controlled documentation-
substitution study on authentic software changes. The accompanying manuscript is
**Semantic Coverage for Release Review: A Controlled Documentation-Substitution
Study on Real Code Changes**, submitted to VFAST Transactions on Software Engineering.
Submission does not imply acceptance or publication.

## What was evaluated

The corpus contains 192 merged pull requests from FastAPI, Apache Airflow,
Home Assistant and Keycloak. They form 32 six-PR clusters and 128 constructed
cases: 32 development and 96 test cases. Each code bundle is paired with its own
descriptions and with a controlled description substitution. Construction labels
identify the intervention and displaced files; they are not human judgments of
semantic completeness.

The development-selected Gemma-3-12B/Nomic pipeline has mean test F1 0.7552 over
three fresh runs, versus 0.6917 for raw-patch Nomic embeddings. Its lower flagged
file/line volume comes with lower case and target-file recall. External transfer
and the 26-pair natural-correction study are reported separately and show limits
to generalization. There is no independent human semantic annotation of the
controlled PR cases and no measured reviewer-time saving.

## Reproduce saved calculations

Python 3.11 or newer is required for the pinned replay environment. The publication
replay was tested with Python 3.12.14 and NumPy 2.3.5. No GPU, LM Studio, model
weights, API key or paid service is needed for the commands below.

```text
python -m pip install -r requirements-replay.txt
python scripts/download_artifacts.py
python scripts/reproduce.py
```

After dependencies are installed, only the download command uses the network.
It fetches the **v1.0.0** release
asset and verifies its SHA-256. To replay entirely offline, obtain the release
asset separately and use:

```text
python scripts/download_artifacts.py --file /path/to/VTSE_2711_PUBLIC_DATA_v1.0.0.zip
python scripts/reproduce.py
```

The wrapper verifies nested archive hashes, extracts primary and external data
into separate directories, and runs calculations with Python network connections
blocked. Outputs go to `replay/`. It checks primary metrics and development
selection, the 23-group/bootstrap sensitivity, natural-case counts/vector scores,
external reference/threshold calculations, and changed-line inspection budgets.
This is replay of saved outputs, not a new model run or independent human validation.

## Repository contents

| Path | Purpose |
|---|---|
| `code/` | Original saved-output audit scripts and external reference calculations |
| `original_pipeline/src/` | Original implementation modules for inspection; not run by default |
| `scripts/` | Portable download, integrity, isolated extraction and offline replay wrappers |
| `LICENSES/`, `THIRD_PARTY_NOTICES.md` | Original upstream terms and bundled-source notices |
| `PUBLICATION_SCOPE.md` | Public-release exclusions and provenance boundaries |
| `PROVENANCE.json` | Exact origins and hashes of exposed research scripts |
| `ARCHIVES_SHA256SUMS.txt` | Hashes of the scientific data ZIPs inside the release asset |

The large scientific ZIPs are release assets, not Git history. Their roots must
remain separate: primary and external archives both contain `replication/`.
The manuscript, editorial response and journal template are absent from this
repository and its release asset.
The public archive's manifest records omitted administrative/template files;
scientific inputs and saved model outputs are preserved. Technical historical
records can contain host-specific paths; use the portable replay commands above.

## New inference and limitations

`original_pipeline/src/` is the archived implementation, not a one-command
portable fresh-generation service. New inference requires the pinned weights,
tokenizers, prompts and recorded LM Studio/runtime settings in the data archive.
Keep new generations separate from saved outputs. A requested seed does not
guarantee deterministic backend behavior.

Controlled substitution does not establish naturally undocumented behavior,
defects, malicious intent, software safety or industrial effectiveness. Natural
targets are author-mapped, and corrected documentation is not a verified negative
class. External snapshots measure a different task. Weak model agreement remains
a result rather than a substitute for human annotation.

## License, citation and assistance

Original research code and documentation use the root MIT license. Third-party
materials retain their own terms; see THIRD_PARTY_NOTICES.md. Cite the tagged
artifact using CITATION.cff. No archival DOI is claimed for this GitHub release.
AI tools assisted with programming, analysis, execution and editing; Sergei
Fedorov is responsible for the scientific decisions and reported artifacts.

## Publication replay check

On 19 October 2026, all seven default replay steps completed successfully on
the public data derivative with Python network connections blocked. See
`validation/SUMMARY.json` and `validation/REPLAY_RECEIPT.json`. This checks the
saved calculations, not new inference or independent human annotation.
