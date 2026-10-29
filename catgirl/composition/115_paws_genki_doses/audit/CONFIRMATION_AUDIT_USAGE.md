# Independent confirmation audit reproduction

The independent numeric audit reads only numeric labels and frozen design files; it never reselects doses. After generating the primary confirmation summaries, run:

```sh
python3 catgirl/composition/115_paws_genki_doses/audit/audit_confirmation.py \
  --labels-dir catgirl/composition/115_paws_genki_doses/numeric_results/confirmation \
  --primary-dir catgirl/composition/115_paws_genki_doses/analysis/confirmation
```

For a local metadata audit with the original private result file present, run:

```sh
python3 catgirl/composition/115_paws_genki_doses/audit/audit_metadata_confirmation.py
```

Both scripts verify the unchanged frozen confirmation hashes. If and only if `results.local/labels_discovery/labels_lock.json` is absent, they resolve that one frozen path to `numeric_results/discovery/labels_lock.json` and verify it against the exact same frozen expected SHA-256. An existing local lock is always checked directly; no other missing path receives a fallback. The published lock has identical bytes. No frozen lock is edited or weakened for public reproduction. The metadata audit still requires the private raw confirmation JSONL and is optional for public numeric reproduction.

Reports and independently reconstructed CSVs are saved under `audit/confirmation/`; discovery audit artifacts remain unchanged.
