# Clean CPU recovery of public numeric results

After both discovery and confirmation exports are present, run from the repository root:

```sh
python3 catgirl/composition/115_paws_genki_doses/verify_cpu_recovery.py
```

This creates a clean temporary tree containing only explicitly listed public design files, the two stdlib analysis scripts, frozen discovery selection, numeric label CSVs and scoring locks. It verifies every public numeric CSV against its scoring-lock SHA256 and checks the export-manifest lock hash when present. It runs both analyses without raw inference, reviewer files, model weights or GPU access. The discovery summary must match byte for byte and every selection field must match, allowing only the scoring-lock path to differ. All five confirmation summary CSVs must match byte for byte.

The compact public report is written to `catgirl/composition/115_paws_genki_doses/CPU_RECOVERY.json`; it contains hashes and checks, with no absolute temporary paths. Recovered analyses live only in the deleted temporary directory. Published analyses and inference outputs are never overwritten. An identical report may be reproduced; use `--report relative/path/CPU_RECOVERY.json` to preserve a changed verification as a separate artifact.
