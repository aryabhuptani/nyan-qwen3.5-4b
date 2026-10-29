# Independent checks and recovery

These helpers use Python's standard library and need no GPU or model weights.
The audit checks point estimates and joins; it cannot validate reviewers' semantic
judgments or justify generalization beyond these prompts.

Before using real data, run synthetic checks:

```sh
python3 catgirl/composition/113_catgirl_circuits/independent_audit.py --selftest
python3 catgirl/composition/113_catgirl_circuits/recovery_bundle.py --selftest
```

After masked review and immutable label locking, first run `analyze.py` as described
in `ANALYSIS_INTERFACE.md`. Independently audit each final and original reviewer
table set. Substitute the actual paths below; the selected label CSV's hash must
appear under its filename in the supplied `labels_lock.json`.

```sh
python3 catgirl/composition/113_catgirl_circuits/independent_audit.py \
  --labels PATH/labels.csv --labels-lock PATH/labels_lock.json \
  --forecasts PATH/forecasts_normalized.csv \
  --baseline-rates PATH/discovery_base_rates.json \
  --primary-dir PATH/analysis --output PATH/independent_audit.json
```

The optional baseline-rates argument must match the primary analysis invocation.
The audit independently recomputes every state-feature rate, five canonical
four-cell pair effects and interactions, and individual/session/arm Brier scores.
Unknown labels retain their missingness and complete-case denominators. Comparison
fails on mismatches exceeding `1e-12`. Bootstrap intervals and sign-flip p-values
remain outside this independent arithmetic check.

Build a compact recovery archive after experiment completion:

```sh
python3 catgirl/composition/113_catgirl_circuits/recovery_bundle.py \
  --output /tmp/study113-recovery.zip
```

It includes all study JSON, JSONL, CSV, Python, Markdown, text, HTML, and image
files, including `results.local` raw replies and achieved intervention summaries,
cases, jobs, dose selection, forecast packets and forecasts, masked reviews,
adjudications, immutable label locks, tables and reports. Historical inference
dependencies and frozen normalization are included under their repository paths.
Model weights, caches, logs, full agent transcripts, and other archives are excluded.
If required review or forecast files live outside the study directory, include each
explicitly with `--extra /path/to/file-or-directory`; these appear under `extra/`.
Unpack any packet-only archives into the study first if their contents are needed.

Verify or extract with:

```sh
python3 catgirl/composition/113_catgirl_circuits/recovery_bundle.py \
  --verify /tmp/study113-recovery.zip --extract /tmp/study113-restored
```

The destination must be empty or absent. Every included file's SHA-256 and byte
count are checked against `RECOVERY_MANIFEST.json` before extraction. From the
restored root, rerun the primary analysis and audit with the restored relative
paths. Original absolute paths embedded in evidence are provenance, and should be
mapped to the corresponding restored files when supplying command arguments.
GPU inference additionally requires installing the recorded MLX dependencies and
obtaining the pinned original BF16 model revision; weights are deliberately absent.
