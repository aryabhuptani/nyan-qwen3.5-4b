# Recovery and independent verification

The public numeric-only labels and instructions in numeric_results/ reproduce all numerical tables without model weights or a GPU. Both original reviews and uncertainty bounds are retained. Semantic review requires the full replies/evidence; their public upload is pending explicit approval.

The local complete archive can be built and verified with standard-library Python:

```sh
python3 catgirl/composition/114_paws_genki/recovery_bundle.py \
  --output /tmp/study114-complete.zip
python3 catgirl/composition/114_paws_genki/recovery_bundle.py \
  --verify /tmp/study114-complete.zip --extract /tmp/study114-restored
```

Use a nonexistent archive and empty/nonexistent extraction directory. SHA-256 and byte counts are checked before extraction. The archive includes all114 source, cases, forecasts, complete raw replies/intervention traces, masked reviews/adjudication, numeric labels and tables, image assets, and the small historical runner dependencies. It excludes weights, caches, logs, full agent transcripts and other archives. Study113 runner/source bytes and numeric selection labels are included to reconstruct the adapter and its provenance.

From the restored root run the numeric_results/README.md commands with restored paths. Repeat for each of the five label tables and compare generated CSV bytes with its archived analysis directory. The independent verifier recomputes all numerical tables, intervals and tests separately and checks the scoring lock through export_manifest.json. CPU_REPLAY.json records the actual clean-directory check and archive hash; it is created after the archived inputs are fixed.

MLX inference additionally needs the recorded environment (Python3.12.16, MLX0.32.4, mlx-lm0.32.2, transformers5.20.2, numpy2.5.4), original BF16 model snapshot and Apple Silicon. All inputs and scripts are pinned by the original freeze; extra auditing, rendering and recovery helpers are later reproducibility additions.
