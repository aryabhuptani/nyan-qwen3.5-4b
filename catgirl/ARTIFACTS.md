# Reproducibility artifacts

All links below are committed in the public [nyan-qwen3.5-4b repository](https://github.com/aryabhuptani/nyan-qwen3.5-4b).

| Purpose | Artifacts |
|---|---|
| Run the model | [Reproduction instructions](REPRODUCE.md), [demo](demo.py), [MLX runner](runner.py), [pinned Python dependencies](requirements.txt) |
| Exact intervention | [Frozen JSON recipe](candidates/catgirl_v1.json), [4,096-row CSV](candidates/catgirl_v1_coordinates.csv), [validation decisions](VALIDATION.json) |
| Evaluate direct induction | [32-case prompts](evaluation_extended.json), [070 job](070_frozen_heldout.json), [raw baseline and candidate outputs](results/070_frozen_heldout.jsonl), [review](results/070_review.json), [CPU constraint scorer](score_heldout_constraints.py) |
| Evaluate deep conversation | [Conversation prompts](conversation_stress.json), [069 job](069_conversation_stress.json), [4-bit raw outputs](results/069_conversation_stress.jsonl), [review](results/069_review.json) |
| Check original BF16 transfer | [071 job](071_bf16_frozen_stress.json), [BF16 runner](public_control_runner.py), [normalization reference](public_qwen35_norms.json), [raw outputs](results/071_bf16_frozen_stress.jsonl), [review](results/071_review.json) |
| Separate startup protocol | [Startup validation script](startup_validation.py), [073 job](073_startup_validation.json), [raw outputs](results/073_startup_validation.jsonl), [review](results/073_review.json) |
| Direct “Who are you?” example | [First-turn raw transcript](results/074_who_are_you.jsonl), [terminal output](results/074_who_are_you.txt) |
| Verify runnable demo | [Actual GPU smoke transcript](results/072_cli_smoke.jsonl), [terminal output](results/072_cli_smoke.txt) |
| Re-run final training stage | [Portable final-stage job](060_reproduce.json), [original training job](060_neural_bootstrap.json), [training examples](discovery_bootstrap.json), [saved warm start](results/052_late_persona.recipes.json), [optimizer](affine_optimizer.py), [final learned recipes](results/060_neural_bootstrap.recipes.json), [training log](results/060_neural_bootstrap.jsonl) |
| Explain and share | [Report](REPORT.md), [post draft](POST_DRAFT.md), [PNG](artifacts/catgirl_evidence.png), [SVG](artifacts/catgirl_evidence.svg) |
| Verify bundle integrity | [SHA-256 manifest](reproducibility_manifest.json), [CPU verification script](verify_artifacts.py) |

From the repository root, run this without installing MLX or downloading weights:

```sh
python3 catgirl/verify_artifacts.py
```

The manifest covers all committed files under `catgirl/` plus the root README, license, and project configuration, excluding the manifest itself. It records model IDs/revisions separately. Model weights are fetched from Hugging Face at the pinned revision, not included in Git. Intermediate activation/gradient NPZ caches are not included; inference and the saved final-stage optimization do not need them. This is an inference reproduction bundle and saved final-stage rerun, not a guarantee of bit-identical retraining of the entire exploratory search.

Raw records intentionally retain original machine paths as provenance. The demo resolves the pinned model on another machine; the portable final-stage job uses repository-relative inputs.
