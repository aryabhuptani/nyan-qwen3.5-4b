"""Frozen catgirl_v1 demo: one neural bootstrap, then unsteered conversation."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT=Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_candidate(path):
    path=Path(path).expanduser().resolve()
    candidate=json.loads(path.read_text())
    condition=candidate['condition']
    specs=condition['neurons']
    if digest(specs)!=candidate['neuron_recipe_sha256'] or digest(condition)!=candidate['condition_sha256']:
        raise ValueError('Frozen candidate hashes do not match its recipe and condition')
    if candidate['model']!='mlx-community/Qwen3.5-4B-4bit' or not candidate.get('revision'):
        raise ValueError('Expected a pinned MLX Qwen3.5-4B 4-bit candidate')
    if (condition.get('early_tokens')!=20 or condition.get('tail_scale')!=0
            or condition.get('prefill_scale')!=0 or not condition.get('bootstrap_first_turn_only')):
        raise ValueError('Expected the frozen first-turn-only bootstrap20 schedule')
    if candidate['protocol']['system_prompt']!='You are a helpful assistant.':
        raise ValueError('Expected the frozen neutral system prompt')
    coordinates=set()
    for spec in specs:
        layer,neuron=spec['layer'],spec['neuron']
        if type(layer) is not int or type(neuron) is not int or not 0<=layer<32 or not 0<=neuron<9216:
            raise ValueError('Invalid frozen neuron coordinate')
        value=spec['factor'] if spec.get('mode')=='mul' else spec['amount']
        if not math.isfinite(value):
            raise ValueError('Frozen recipe contains a nonfinite factor or activation amount')
        coordinates.add((layer,neuron))
    if len(coordinates)!=candidate['unique_neuron_coordinates'] or len(specs)!=candidate['intervention_specs']:
        raise ValueError('Frozen neuron/specification counts do not match the recipe')
    return path,candidate


def resolve_model(candidate,cache_dir=None):
    local=Path(candidate.get('loaded_model_path',''))
    if local.is_dir() and local.name==candidate['revision'] and (local/'config.json').exists():
        return str(local.resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=candidate['model'],revision=candidate['revision'],cache_dir=cache_dir)


class Conversation:
    def __init__(self,runner,candidate,baseline=False):
        self.runner=runner
        self.candidate=candidate
        self.baseline=baseline
        decoding=candidate['decoding']
        runner.repetition_penalty=decoding['repetition_penalty']
        runner.repetition_context_size=decoding['repetition_context_size']
        self.reset()

    def reset(self):
        self.messages=[{'role':'system','content':self.candidate['protocol']['system_prompt']}]
        self.turn=0
        self.runner.steer([])

    def reply(self,prompt,max_tokens):
        bootstrap=self.turn==0 and not self.baseline
        recipe=self.candidate['condition']['neurons'] if bootstrap else []
        self.runner.steer(recipe)
        self.runner.early_tokens=20 if bootstrap else None
        self.runner.tail_scale=0.
        self.runner.prefill_scale=0.
        for mlp in self.runner.mlps:
            mlp.position_mask=None
            mlp.capture=False
            mlp.captured=None
        messages=self.messages+[{'role':'user','content':prompt}]
        decoding=self.candidate['decoding']
        try:
            result=self.runner.generate(messages,max_tokens,decoding['seed'],decoding['temperature'])
        finally:
            # No intervention remains active while waiting for another turn.
            self.runner.steer([])
        self.turn+=1
        self.messages=messages+[{'role':'assistant','content':result['text']}]
        return dict(kind='turn',turn=self.turn,messages=messages,
                    effective_intervention_specs=len(recipe),effective_recipe_sha256=digest(recipe),
                    intervention_scope=dict(early_tokens=20 if bootstrap else None,
                                            tail_scale=0.,prefill_scale=0.,
                                            generation='first 20 prediction positions' if bootstrap else 'off',
                                            prefill=False,
                                            boundary_note='The last prompt-template token predicts the first answer token and is included in the 20 positions.'),
                    max_new_tokens=max_tokens,**result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',default=str(ROOT/'candidates'/'catgirl_v1.json'))
    parser.add_argument('--check-only',action='store_true',help='Validate the frozen artifact without loading MLX or a model')
    parser.add_argument('--prompt',action='append',help='Conversation turn; repeat this option for later turns. Omit for interactive chat.')
    parser.add_argument('--baseline',action='store_true',help='Matched neutral model with steering disabled on every turn')
    parser.add_argument('--max-tokens',type=int,default=384)
    parser.add_argument('--transcript',type=Path,help='Create a new JSONL transcript; an existing file is never overwritten')
    parser.add_argument('--cache-dir',help='Hugging Face cache directory when downloading the pinned snapshot')
    args=parser.parse_args()
    if args.max_tokens<1:
        parser.error('--max-tokens must be positive')
    path,candidate=load_candidate(args.candidate)
    if args.check_only:
        print(json.dumps(dict(candidate=str(path),model=candidate['model'],revision=candidate['revision'],
                              unique_neuron_coordinates=candidate['unique_neuron_coordinates'],
                              intervention_specs=candidate['intervention_specs'],
                              neuron_recipe_sha256=candidate['neuron_recipe_sha256'],valid=True),indent=2))
        return
    if args.transcript is not None and args.transcript.exists():
        parser.error('--transcript already exists; choose a new output path')
    model_path=resolve_model(candidate,args.cache_dir)
    # Keep --check-only and --help CPU-only, with no MLX import side effects.
    try:
        from .runner import Runner
    except ImportError:
        from runner import Runner
    (ROOT/'results').mkdir(parents=True,exist_ok=True)
    with redirect_stdout(sys.stderr):
        runner=Runner(model_path)
    if len(runner.layers)!=32 or runner.config.get('text_config',runner.config).get('intermediate_size')!=9216:
        raise ValueError('Loaded model architecture differs from the frozen Qwen3.5-4B recipe')
    if any(getattr(getattr(mlp,name),'bits',None)!=4 for mlp in runner.mlps for name in ('gate_proj','up_proj','down_proj')):
        raise ValueError('Frozen demo requires the pinned 4-bit MLX weights')
    conversation=Conversation(runner,candidate,args.baseline)
    handle=None
    try:
        if args.transcript is not None:
            args.transcript.parent.mkdir(parents=True,exist_ok=True)
            handle=args.transcript.open('x')
            handle.write(json.dumps(dict(kind='session',candidate_file=str(path),
                candidate_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                neuron_recipe_sha256=candidate['neuron_recipe_sha256'],model=candidate['model'],
                revision=candidate['revision'],loaded_model_path=model_path,baseline=args.baseline,
                system_prompt=candidate['protocol']['system_prompt'],decoding=candidate['decoding'],
                model_metadata=runner.metadata))+'\n')
            handle.flush()
        prompts=iter(args.prompt) if args.prompt is not None else None
        if prompts is None:
            print('First reply: neural bootstrap20. Later replies: steering off.' if not args.baseline else 'Baseline: steering off on every reply.')
            print('Type /reset for a new conversation or /quit to leave.')
        while True:
            if prompts is None:
                try:
                    prompt=input('You> ')
                except (EOFError,KeyboardInterrupt):
                    print();break
                if prompt.strip()=='/quit':
                    break
                if prompt.strip()=='/reset':
                    conversation.reset()
                    if handle:
                        handle.write(json.dumps(dict(kind='conversation_reset'))+'\n');handle.flush()
                    continue
            else:
                prompt=next(prompts,None)
                if prompt is None:
                    break
                print('You> '+prompt)
            if not prompt.strip():
                continue
            record=conversation.reply(prompt,args.max_tokens)
            print('Assistant> '+record['text'])
            if handle:
                handle.write(json.dumps(record,ensure_ascii=False)+'\n');handle.flush()
    finally:
        runner.steer([])
        if handle:
            handle.close()


if __name__=='__main__':
    main()
