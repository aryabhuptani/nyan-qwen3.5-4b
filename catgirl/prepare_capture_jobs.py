"""Generate a CPU-only screen using raw assistant-activation contrasts.

Run with the project virtualenv's Python. The job stays outside jobs/ so a
running GPU watcher cannot start it before it has been reviewed.
"""
import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def prepare(source, destination):
    with np.load(source) as archive:
        values = archive["values"].astype(np.float32)
    if values.ndim != 4 or values.shape[1] != 2 or values.shape[2] < 28:
        raise ValueError("Expected [examples, 2, layers >= 28, neurons] values")
    if not np.isfinite(values).all():
        raise ValueError("Capture contains nonfinite activation values")

    difference = values[:, 1] - values[:, 0]
    mean = difference.mean(axis=0)
    consistency = (np.sign(difference) == np.sign(mean)[None]).mean(axis=0)
    # Exactly the score used by runner.capture, recomputed across every neuron
    # so layer-restricted rankings do not depend on its top-2048 truncation.
    scores = {
        "standardized": np.abs(mean) / (difference.std(axis=0) + .05)
        * consistency ** 3,
        "magnitude": np.abs(mean) * consistency ** 3,
    }
    selections = {}

    def select(ranking, lo, hi, count):
        key = (ranking, lo, hi, count)
        if key not in selections:
            score = scores[ranking][lo:hi + 1]
            order = np.argsort(-score.ravel(), kind="stable")[:count]
            rows, neurons = np.unravel_index(order, score.shape)
            selections[key] = [(int(row + lo), int(neuron))
                               for row, neuron in zip(rows, neurons)]
        return selections[key]

    conditions = [{"id": "baseline", "neurons": []}]
    seen = set()

    def add(ranking, lo, hi, count, factor):
        selected = select(ranking, lo, hi, count)
        # Exact duplicate selections at the same factor need only one screen.
        signature = (factor, tuple(sorted(selected)))
        if signature in seen:
            return
        seen.add(signature)
        conditions.append({
            "id": f"raw_{ranking}_L{lo}-{hi}_K{count}_x{factor}",
            "neurons": [
                {"layer": layer, "neuron": neuron,
                 "amount": float(factor * mean[layer, neuron])}
                for layer, neuron in selected
            ],
        })

    for lo, hi in [(8, 27), (16, 27)]:
        for count in [32, 128, 512]:
            for factor in [1, 3, 10]:
                add("standardized", lo, hi, count, factor)
    # A smaller second-ranker screen keeps the total at 24 conditions while
    # testing whether the standardized ranking favors tiny, stable changes.
    for count in [32, 128, 512]:
        add("magnitude", 8, 27, count, 3)
    for factor in [1, 3]:
        add("magnitude", 16, 27, 128, factor)

    if len(conditions) > 24:
        raise AssertionError("Screen exceeds the condition budget")
    original = json.loads((ROOT / "jobs" / "001_lens_screen.done").read_text())
    job = {
        "conditions": conditions,
        "max_tokens": 128,
        "prompts": original["prompts"],
        "metadata": {
            "source": str(source),
            "examples": int(values.shape[0]),
            "intervention": "Raw additive SwiGLU activation change; no residual-norm normalization",
            "amount": "factor * mean(catgirl_activation - neutral_activation)",
            "standardized_score": "abs(mean_delta) / (std_delta + 0.05) * sign_consistency**3",
            "magnitude_score": "abs(mean_delta) * sign_consistency**3",
            "layer_bounds": "Inclusive, zero based",
        },
    }
    destination.write_text(json.dumps(job, indent=2) + "\n")
    print(f"Wrote {len(conditions)} conditions to {destination}")
    for condition in conditions:
        amounts = np.array([n["amount"] for n in condition["neurons"]])
        print(condition["id"], "neurons=", len(amounts),
              "activation_l2=", float(np.linalg.norm(amounts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "results" / "002_capture.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "006_raw_contrast.json")
    arguments = parser.parse_args()
    prepare(arguments.source, arguments.output)
