"""CPU integration checks for evaluation and demo per-layer timing consumers."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from catgirl import final_evaluation
from catgirl.composition import demo
from catgirl.test_layer_schedules import fake_runner


class ScheduleConsumerTests(unittest.TestCase):
    def setUp(self):
        self.override = {'22': {'early_tokens': 8, 'prefill_scale': 1.0, 'tail_scale': 0.0}}
        self.condition = {'id': 'mixed', 'neurons': [], 'early_tokens': None,
                          'prefill_scale': 0.0, 'tail_scale': 0.0,
                          'layer_schedules': self.override}
        self.protocol = dict(seed=42, temperature=0, repetition_penalty=1.1,
                             repetition_context_size=128)
        self.battery = {'usage': {'system_prompt': 'You are a helpful assistant.',
                                 'decoding': {'seed': 42, 'temperature': 0}},
                        'cases': [{'id': 'short', 'prompt': 'Neutral question.', 'max_new_tokens': 8}],
                        'persistence': {'multiturn': {'id': 'conversation', 'max_new_tokens_per_turn': 8,
                                                     'turns': [{'id': 'turn1', 'prompt': 'First question.'},
                                                               {'id': 'turn2', 'prompt': 'Next question.'}]}}}

    def runner(self):
        runner, snapshots, namespace = fake_runner()
        runner.config = {}
        runner.tok = SimpleNamespace(encode=lambda text, **kwargs: list(range(len(text))),
                                     decode=lambda ids: 'test'[:len(ids)])
        return runner, snapshots, namespace

    def test_evaluation_scope_reset_off_turns_history_and_resume(self):
        runner, snapshots, _ = self.runner()
        condition = dict(self.condition, bootstrap_first_turn_only=True)
        job = dict(conditions=[condition, {'id': 'baseline', 'neurons': []}],
                   include_baseline=False, sections=['short', 'multiturn'],
                   battery_file='battery.json', **self.protocol)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            (root / 'battery.json').write_text(json.dumps(self.battery))
            output = root / 'result.jsonl'
            with patch.object(final_evaluation, 'ROOT', root):
                final_evaluation.evaluate(runner, job, output)
                records = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(len(records), 6)
                self.assertEqual(records[0]['layer_schedules'], self.override)
                self.assertEqual(records[0]['intervention_scope']['layer_schedules'], self.override)
                self.assertEqual(records[1]['layer_schedules'], self.override)
                self.assertEqual(records[2]['layer_schedules'], {})
                self.assertEqual(records[2]['intervention_scope']['generation'], 'off')
                self.assertEqual(records[2]['messages'][-2], {'role': 'assistant', 'content': 'test'})
                self.assertTrue(all(record['layer_schedules'] == {} for record in records[3:]))
                self.assertEqual(snapshots[0][22][3:], (8, 1.0, 0.0))
                self.assertTrue(all(row[3:] == (None, 0.0, 0.0) for row in snapshots[2]))
                self.assertEqual(runner.layer_schedules, {})
                final_evaluation.evaluate(runner, job, output)
                self.assertEqual(len(snapshots), 6)
                # A record with unchanged condition hash but altered effective scope must fail too.
                records[0]['layer_schedules']['22']['early_tokens'] = 9
                output.write_text(''.join(json.dumps(record) + '\n' for record in records))
                with self.assertRaisesRegex(ValueError, 'Resume configuration'):
                    final_evaluation.evaluate(runner, job, output)
                self.assertEqual(runner.layer_schedules, {})

    def test_evaluation_restores_state_after_generation_error(self):
        runner, _, namespace = self.runner()
        prior = {'7': {'early_tokens': 2, 'prefill_scale': 0.0, 'tail_scale': 0.0}}
        runner.layer_schedules = copy.deepcopy(prior)
        def fail(*args, **kwargs):
            raise RuntimeError('fake failure')
        namespace['stream_generate'] = fail
        job = dict(conditions=[self.condition], include_baseline=False,
                   sections=['short'], battery_file='battery.json')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'battery.json').write_text(json.dumps(self.battery))
            with patch.object(final_evaluation, 'ROOT', root):
                with self.assertRaisesRegex(RuntimeError, 'fake failure'):
                    final_evaluation.evaluate(runner, job, root / 'results.jsonl')
        self.assertEqual(runner.layer_schedules, prior)

    def test_demo_logs_override_then_baseline_resets_it(self):
        runner, snapshots, _ = self.runner()
        history = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        record = demo.generate_turn(runner, history, 'Neutral question.', [],
                                    self.condition, self.protocol, 8)
        self.assertEqual(record['effective_scope']['layer_schedules'], self.override)
        self.assertFalse(record['effective_scope']['all_positions'])
        self.assertEqual(snapshots[0][22][3:], (8, 1.0, 0.0))
        self.assertEqual(runner.layer_schedules, {})
        baseline = dict(self.condition, id='baseline')
        baseline.pop('layer_schedules')
        next_record = demo.generate_turn(runner, history, 'Follow-up.', [], baseline, self.protocol, 8)
        self.assertNotIn('layer_schedules', next_record['effective_scope'])
        self.assertTrue(all(row[3:] == (None, 0.0, 0.0) for row in snapshots[1]))
        self.assertEqual(next_record['messages'][-2], {'role': 'assistant', 'content': 'test'})

    def test_demo_failure_clears_override_without_inventing_history(self):
        runner, _, namespace = self.runner()
        history = []
        def fail(*args, **kwargs):
            raise RuntimeError('fake failure')
        namespace['stream_generate'] = fail
        with self.assertRaisesRegex(RuntimeError, 'fake failure'):
            demo.generate_turn(runner, history, 'Neutral question.', [],
                               self.condition, self.protocol, 8)
        self.assertEqual(runner.layer_schedules, {})
        self.assertEqual(history, [])


if __name__ == '__main__':
    unittest.main()
