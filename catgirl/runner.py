"""Single-process Apple GPU neuron intervention runner. Jobs and outputs are JSON.

Indices refer to silu(gate_proj(x))*up_proj(x), before down_proj, zero based.
Additive dose units match neuron-explorer: median layer-output residual norm
divided by the down-projection column norm. Calibration is saved verbatim.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
import numpy as np

ROOT = Path(__file__).resolve().parent


class SteeredMLP(nn.Module):
    def __init__(self, original, layer):
        super().__init__()
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj
        self.layer = layer
        self.delta = None
        self.capture = False
        self.start = 0
        self.captured = None

    def __call__(self, x):
        h = nn.silu(self.gate_proj(x)) * self.up_proj(x)
        if self.capture:
            self.captured = h[:, self.start:, :].astype(mx.float32).mean(axis=(0, 1))
        if self.delta is not None:
            h = h + self.delta.astype(h.dtype)
        return self.down_proj(h)


class Runner:
    def __init__(self, model_path):
        start = time.time()
        self.model, self.tok, self.config = load(model_path, return_config=True)
        self.layers = self.model.layers
        self.mlps = []
        for i, layer in enumerate(self.layers):
            layer.mlp = SteeredMLP(layer.mlp, i)
            self.mlps.append(layer.mlp)
        self.residual = {}
        self.columns = {}
        self.metadata = dict(model=model_path, config=self.config,
                             versions={p: importlib.metadata.version(p) for p in ['mlx','mlx-lm','transformers']},
                             device=mx.device_info(), loaded_seconds=time.time()-start)
        print(json.dumps(dict(event='loaded', **self.metadata)), flush=True)
        self.calibrate()

    def tokens(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)

    def calibrate(self):
        # Capture actual residual leaving each layer on neutral reference text.
        klass = type(self.layers[0])
        original = klass.__call__
        identities = {id(layer): i for i, layer in enumerate(self.layers)}
        values = {}
        def call(layer, *args, **kwargs):
            result = original(layer, *args, **kwargs)
            values[identities[id(layer)]] = mx.linalg.norm(result.astype(mx.float32), axis=-1)
            return result
        klass.__call__ = call
        reference = 'The history of the region is long and varied. Farmers settled the river valley first, then traders arrived. A careful engineer checks assumptions and tests a small example. The children enjoyed their afternoon by the river. To sort a list, compare its elements and arrange them in order. '
        try:
            out = self.model(mx.array(self.tokens([{'role':'user','content':reference * 3}]))[None])
            mx.eval(out, values)
            self.residual = {i: float(np.median(np.array(v))) for i,v in values.items()}
        finally:
            klass.__call__ = original
        (ROOT/'results'/'calibration.json').write_text(json.dumps(dict(metadata=self.metadata,residual_norms=self.residual,reference=reference*3),indent=2))
        print(json.dumps(dict(event='calibrated', residual=self.residual)), flush=True)

    def column(self, layer, neuron):
        key = (layer, neuron)
        if key not in self.columns:
            proj = self.mlps[layer].down_proj
            if hasattr(proj, 'scales'):
                w = mx.dequantize(proj.weight, proj.scales, getattr(proj,'biases',None), group_size=proj.group_size, bits=proj.bits, mode=getattr(proj,'mode','affine'))
            else:
                w = proj.weight
            c = w[:, neuron].astype(mx.float32)
            mx.eval(c)
            self.columns[key] = c
        return self.columns[key]

    def steer(self, recipe):
        for mlp in self.mlps:
            mlp.delta = None
        grouped = {}
        for spec in recipe:
            layer, neuron = spec['layer'], spec['neuron']
            if 'amount' in spec:
                amount=spec['amount']
            else:
                column = self.column(layer, neuron)
                amount = spec['dose']*self.residual[layer]/float(mx.linalg.norm(column))
            grouped.setdefault(layer, []).append((neuron, amount))
        for layer, pairs in grouped.items():
            delta = np.zeros(self.config.get('text_config', self.config)['intermediate_size'], dtype=np.float32)
            for neuron, amount in pairs:
                delta[neuron] += amount
            self.mlps[layer].delta = mx.array(delta)
        mx.eval([m.delta for m in self.mlps if m.delta is not None])

    def generate(self, messages, max_tokens, seed=0, temperature=0):
        mx.random.seed(seed)
        ids = self.tokens(messages)
        started = time.time()
        chunks=[]
        last=None
        for part in stream_generate(self.model,self.tok,ids,max_tokens=max_tokens,sampler=make_sampler(temperature)):
            chunks.append(part.text)
            last=part
        return dict(text=''.join(chunks), prompt_tokens=len(ids), generated_tokens=last.generation_tokens,
                    generation_tps=last.generation_tps, prefill_tps=last.prompt_tps,
                    finish_reason=last.finish_reason, seconds=time.time()-started)

    def run(self, job, output):
        if job.get('action') in ['capture', 'capture_aligned']:
            return self.capture(job, output)
        if job.get('action') == 'optimize':
            from gradient_search import optimize
            return optimize(self,job,output)
        if job.get('action') == 'score':
            return self.score(job,output)
        completed = set()
        if output.exists():
            for line in output.read_text().splitlines():
                r = json.loads(line)
                completed.add((r['condition'], r['prompt_id']))
        with output.open('a') as f:
            for condition in job['conditions']:
                self.steer(condition.get('neurons',[]))
                for p in job['prompts']:
                    if (condition['id'], p['id']) in completed:
                        continue
                    messages=p.get('messages',[{'role':'user','content':p.get('prompt','')}])
                    result=self.generate(messages,p.get('max_tokens',job.get('max_tokens',128)),job.get('seed',0),job.get('temperature',0))
                    record=dict(condition=condition['id'],recipe=condition.get('neurons',[]),prompt_id=p['id'],messages=messages,**result)
                    f.write(json.dumps(record,ensure_ascii=False)+'\n'); f.flush()
                    print(json.dumps({k:v for k,v in record.items() if k not in ['recipe','messages']},ensure_ascii=False),flush=True)

    def capture(self, job, output):
        self.steer([])
        data=json.loads((ROOT/'discovery.json').read_text())
        captures=[]
        for pair in data['pairs']:
            poles=[]
            for pole in ['neutral','catgirl']:
                # Use the same neutral system context for both continuations.
                # Only assistant positions are averaged; cue words in the system
                # prompt therefore cannot directly dominate the contrast.
                aligned=job.get('action')=='capture_aligned'
                system=data[pole+'_system'] if aligned else data['neutral_system']
                messages=[{'role':'system','content':system}, {'role':'user','content':pair['prompt']}]
                prefix=self.tokens(messages)
                continuation=self.tok.encode(pair['neutral' if aligned else pole], add_special_tokens=False)
                for m in self.mlps:
                    m.capture=True; m.start=len(prefix)
                logits=self.model(mx.array(prefix+continuation)[None])
                mx.eval(logits, [m.captured for m in self.mlps])
                poles.append(np.stack([np.array(m.captured) for m in self.mlps]))
                for m in self.mlps:
                    m.capture=False; m.captured=None
            captures.append(poles)
            print(json.dumps({'event':'captured','pair':pair['id']}),flush=True)
        values=np.array(captures)
        np.savez_compressed(output.with_suffix('.npz'),values=values)
        diff=values[:,1]-values[:,0]
        mean=diff.mean(0)
        score=np.abs(mean)/(diff.std(0)+.05)
        consistency=(np.sign(diff)==np.sign(mean)[None]).mean(0)
        score*=consistency**3
        ranking=[]
        for flat in np.argsort(score.ravel())[::-1][:2048]:
            l,n=np.unravel_index(flat,score.shape)
            ranking.append({'layer':int(l),'neuron':int(n),'mean_delta':float(mean[l,n]),'score':float(score[l,n]),'consistency':float(consistency[l,n])})
        output.with_suffix('.ranking.json').write_text(json.dumps(ranking,indent=2))

    def score(self,job,output):
        data=json.loads((ROOT/'discovery.json').read_text())
        with output.open('a') as f:
            for condition in job['conditions']:
                self.steer(condition.get('neurons',[]))
                scores=[]
                for pair in data['pairs'][:job.get('examples',4)]:
                    prefix=self.tokens([{'role':'user','content':pair['prompt']}])
                    values={}
                    for pole in ['neutral','catgirl']:
                        target=self.tok.encode(pair[pole],add_special_tokens=False)
                        logits=self.model(mx.array(prefix+target)[None])[:,len(prefix)-1:-1].astype(mx.float32)
                        logp=logits-mx.logsumexp(logits,axis=-1,keepdims=True)
                        loss=mx.take_along_axis(logp,mx.array(target)[None,:,None],axis=-1).mean()
                        values[pole]=float(loss)
                    scores.append(values)
                record={'condition':condition['id'],'recipe':condition.get('neurons',[]),'scores':scores,'margin':float(np.mean([v['catgirl']-v['neutral'] for v in scores]))}
                f.write(json.dumps(record)+'\n');f.flush()
                print(json.dumps(record),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='mlx-community/Qwen3.5-4B-4bit')
    parser.add_argument('--job')
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    runner=Runner(args.model)
    if args.job:
        job=Path(args.job)
        runner.run(json.loads(job.read_text()),ROOT/'results'/(job.stem+'.jsonl'))
    if args.watch:
        print('WATCHING',flush=True)
        while True:
            for path in sorted((ROOT/'jobs').glob('*.json')):
                active=path.with_suffix('.running')
                path.rename(active)
                try:
                    runner.run(json.loads(active.read_text()),ROOT/'results'/(path.stem+'.jsonl'))
                    active.rename(path.with_suffix('.done'))
                except Exception:
                    traceback.print_exc()
                    active.rename(path.with_suffix('.failed'))
            time.sleep(1)


if __name__=='__main__':
    main()
