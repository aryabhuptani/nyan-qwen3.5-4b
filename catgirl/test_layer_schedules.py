"""CPU fakes exercise actual Runner methods without importing MLX/model weights."""
import ast
import copy
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from catgirl.layer_schedules import normalize_layer_schedules
from catgirl.composition_worker import validate_job


def fake_runner():
    source = Path(__file__).with_name('runner.py')
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Runner')
    snapshots = []

    def stream(model, tokenizer, ids, **kwargs):
        snapshots.append([
            (m.layer, m.seen, m.boundary, m.early_tokens, m.prefill_scale, m.tail_scale)
            for m in runner.mlps
        ])
        yield SimpleNamespace(text='test', generation_tokens=1, generation_tps=1,
                              prompt_tps=1, finish_reason='stop')

    namespace = dict(normalize_layer_schedules=normalize_layer_schedules,
                     mx=SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None)),
                     np=SimpleNamespace(isfinite=math.isfinite), stream_generate=stream,
                     make_sampler=lambda temperature: None,
                     make_logits_processors=lambda **kwargs: [], time=time, json=json)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'), namespace)
    runner = namespace['Runner'].__new__(namespace['Runner'])
    runner.mlps = [SimpleNamespace(layer=i) for i in range(32)]
    runner.tokens = lambda messages: [1, 2, 3, 4, 5]
    runner.steer = lambda recipe: None
    runner.model = runner.tok = None
    runner.metadata = {'model': 'cpu-fake', 'calibration_file': 'cpu-fake'}
    runner.layer_schedules = {}
    return runner, snapshots, namespace


class LayerScheduleTests(unittest.TestCase):
    def setUp(self):
        self.override = {'22': {'early_tokens': 8, 'prefill_scale': 1, 'tail_scale': 0}}
        self.job = {'conditions': [
            {'id': 'mixed', 'neurons': [], 'early_tokens': None,
             'layer_schedules': self.override},
            {'id': 'plain', 'neurons': [], 'early_tokens': 16,
             'prefill_scale': 0.25, 'tail_scale': 0.5}],
            'prompts': [{'id': 'question', 'prompt': 'Neutral question.'}],
            'repetition_penalty': 1.1}

    def test_generate_assigns_each_layer_override_and_global_fallback(self):
        runner, snapshots, _ = fake_runner()
        runner.early_tokens, runner.prefill_scale, runner.tail_scale = 16, 0.25, 0.5
        runner.layer_schedules = self.override
        runner.generate([], 1)
        for layer, seen, boundary, early, prefill, tail in snapshots[0]:
            self.assertEqual((seen, boundary), (0, 4))
            self.assertEqual((early, prefill, tail),
                             (8, 1.0, 0.0) if layer == 22 else (16, 0.25, 0.5))

    def test_conditions_reset_override_and_log_exact_identity(self):
        runner, snapshots, _ = fake_runner()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results.jsonl'
            runner.run(self.job, output)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(records[0]['layer_schedules'], self.override)
            self.assertEqual(records[1]['layer_schedules'], {})
            self.assertEqual(snapshots[0][22][3:], (8, 1.0, 0.0))
            self.assertTrue(all(row[3:] == (None, 0.0, 0.0)
                                for row in snapshots[0] if row[0] != 22))
            self.assertTrue(all(row[3:] == (16, 0.25, 0.5) for row in snapshots[1]))
            self.assertEqual(runner.layer_schedules, {})
            runner.run(self.job, output)
            self.assertEqual(len(snapshots), 2)  # Identical resume does not generate again.
            altered = copy.deepcopy(self.job)
            altered['conditions'][0]['layer_schedules']['22']['early_tokens'] = 9
            with self.assertRaisesRegex(ValueError, 'Resume would mix'):
                runner.run(altered, output)
            self.assertEqual(runner.layer_schedules, {})

    def test_legacy_resume_without_mapping_matches_default_only(self):
        runner, _, _ = fake_runner()
        job = copy.deepcopy(self.job)
        job['conditions'] = [job['conditions'][1]]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results.jsonl'
            runner.run(job, output)
            record = json.loads(output.read_text())
            record.pop('layer_schedules')
            output.write_text(json.dumps(record) + '\n')
            runner.run(job, output)
            job['conditions'][0]['layer_schedules'] = self.override
            with self.assertRaisesRegex(ValueError, 'Resume would mix'):
                runner.run(job, output)

    def test_failed_generation_clears_mapping(self):
        runner, _, namespace = fake_runner()
        def fail(*args, **kwargs):
            raise RuntimeError('fake generation failed')
        namespace['stream_generate'] = fail
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'fake generation failed'):
                runner.run(self.job, Path(directory) / 'results.jsonl')
        self.assertEqual(runner.layer_schedules, {})

    def test_worker_rejects_invalid_or_unsupported_schedules(self):
        for mapping in [[], {'32': self.override['22']},
                        {'22': {'early_tokens': 8}},
                        {'22': dict(self.override['22'], early_tokens=True)},
                        {'22': dict(self.override['22'], tail_scale=float('nan'))}]:
            job = copy.deepcopy(self.job)
            job['conditions'][0]['layer_schedules'] = mapping
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                validate_job(job)
        self.assertEqual(validate_job(self.job)['conditions'][0]['layer_schedules'], self.override)
        job = copy.deepcopy(self.job)
        job['action'] = 'evaluate'
        with self.assertRaisesRegex(ValueError, 'final_evaluation'):
            validate_job(job)


if __name__ == '__main__':
    unittest.main()
