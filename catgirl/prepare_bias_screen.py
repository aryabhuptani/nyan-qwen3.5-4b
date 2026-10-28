"""Prepare bias-only discovery screens on CPU, outside the watcher queue.

Default outputs: 047_bias_screen.json (candidate, eight prompts) and
047a_bias_deep_baseline.json (unsteered, the two new deep prompts only).
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(recipes, output, baseline_output, model=None):
    prompt_source = ROOT / "042_affine_screen.json"
    previous = json.loads(prompt_source.read_text())
    calibration = ROOT / "results" / "calibration.json"
    model = model or json.loads(calibration.read_text())["metadata"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    system = "You are a helpful assistant."

    def messages(prompt):
        return [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]

    def count(history):
        encoded = tokenizer.apply_chat_template(
            history, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False,
        )
        return len(encoded["input_ids"]) if hasattr(encoded, "keys") else len(encoded)

    guide = {
        "id": "bias_weekend_nature_photography",
        "messages": messages(
            "Write a practical guide to a weekend of nature photography for a beginner. "
            "Use exactly six numbered sections, with 80 to 100 words in each section. "
            "Cover planning a nearby outing, preparing a phone or camera, working with "
            "natural light, composing images, photographing wildlife respectfully, and "
            "reviewing and backing up photographs afterward. Include low-cost options "
            "and concrete actions throughout."
        ),
        "max_tokens": 1024,
        "metadata": {
            "quality_checks": [
                "Six numbered sections with 80 to 100 words per section",
                "All six requested photography topics covered accurately",
                "Practical low-cost advice and respectful wildlife behavior",
                "Coherent complete answer without repetitive loops",
            ],
            "persona_review": "Review feline identity and mannerisms in each output third; isolated opening cues do not demonstrate persistence.",
            "source": "New discovery prompt; not a held-out evaluation case",
        },
    }

    seed = 2026102847
    generator = random.Random(seed)
    code = "".join(generator.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(8))
    target_note = 927
    target_line = f"Survey note {target_note}: the reference code is {code}."
    header = "Read this neutral field-survey log carefully before answering the question at the end.\n\n"
    question = (
        f"\n\nWhat reference code appears in survey note {target_note}? "
        "Then explain how to organize a shared collection of outdoor equipment: "
        "give three practical paragraphs about labeling, checking condition, and "
        "returning borrowed items."
    )
    equipment = ["measuring tapes", "sample trays", "clipboards", "rain covers", "markers", "storage sleeves"]
    lines = []
    for index in range(6000, 8000):
        lines.append(
            f"Survey note {index}: station {index % 29 + 1} contains "
            f"{generator.randrange(2, 48)} {equipment[index % len(equipment)]}; "
            "the routine check is recorded."
        )
        middle = len(lines) // 2
        text = header + "\n".join(lines[:middle] + [target_line] + lines[middle:]) + question
        if count(messages(text)) > 8192:
            lines.pop()
            break
    middle = len(lines) // 2
    text = header + "\n".join(lines[:middle] + [target_line] + lines[middle:]) + question
    archive = {
        "id": "bias_survey_archive8192", "messages": messages(text), "max_tokens": 768,
        "metadata": {
            "source": "New deterministic discovery archive; not a held-out evaluation case",
            "filler_seed": seed, "target_input_tokens": 8192,
            "record_count": len(lines) + 1, "target_record": target_note,
            "target_record_fraction": middle / (len(lines) + 1), "reference_code": code,
            "target_character_offset": text.index(target_line),
            "quality_checks": [f"Retrieves {code}", "Three useful paragraphs covering labels, condition checks, and returns"],
        },
    }
    new_prompts = [guide, archive]
    for prompt in new_prompts:
        prompt["metadata"]["actual_input_tokens"] = count(prompt["messages"])

    saved = json.loads(recipes.read_text())
    conditions = saved.get("conditions", [saved.get("condition", saved)])
    candidates = [condition for condition in conditions if condition.get("neurons")]
    if not candidates:
        raise ValueError("No nonempty neuron recipe found")
    selected = max(candidates, key=lambda condition: len({(n["layer"], n["neuron"]) for n in condition["neurons"]}))
    candidate = copy.deepcopy(selected)
    candidate["id"] = "bias_only_strength1.0"
    candidate["repetition_penalty"] = 1.1
    candidate["repetition_context_size"] = 128
    decoding = {key: copy.deepcopy(previous[key]) for key in (
        "max_tokens", "seed", "temperature", "repetition_penalty", "repetition_context_size"
    ) if key in previous}
    decoding.update(repetition_penalty=1.1, repetition_context_size=128)
    provenance = {
        "recipe_source": str(recipes.resolve()), "recipe_sha256": digest(recipes),
        "source_condition": selected.get("id"), "strength": 1.0,
        "existing_prompt_source": str(prompt_source), "existing_prompt_sha256": digest(prompt_source),
        "tokenizer_model": str(model), "new_prompt_source": "prepare_bias_screen.py deterministic discovery additions",
        "baseline_job": str(baseline_output.resolve()),
        "existing_six_baseline_source": "catgirl/results/042_affine_screen.jsonl; condition affine_baseline",
    }
    job = dict(decoding, conditions=[candidate],
               prompts=copy.deepcopy(previous["prompts"]) + new_prompts,
               metadata=provenance)
    baseline = dict(decoding, conditions=[dict(id="bias_deep_baseline", neurons=[],
                    repetition_penalty=1.1, repetition_context_size=128)],
                    prompts=copy.deepcopy(new_prompts), metadata=provenance)
    output.write_text(json.dumps(job, indent=2, ensure_ascii=False) + "\n")
    baseline_output.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n")
    print(f"Candidate: {len(job['prompts'])} prompts -> {output}")
    print(f"New deep baseline: 2 prompts -> {baseline_output}")
    print(f"Archive input: {archive['metadata']['actual_input_tokens']} tokens; reference code {code}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, default=ROOT / "results" / "046_bias_only_train.recipes.json")
    parser.add_argument("--output", type=Path, default=ROOT / "047_bias_screen.json")
    parser.add_argument("--baseline-output", type=Path, default=ROOT / "047a_bias_deep_baseline.json")
    parser.add_argument("--model", help="Cached tokenizer path; defaults to calibration metadata")
    args = parser.parse_args()
    prepare(args.recipes, args.output, args.baseline_output, args.model)
