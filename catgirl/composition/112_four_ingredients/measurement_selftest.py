"""CPU-only checks, including no import/load of MLX."""
import sys
import numpy as np
from measure import apply_numpy, segmented_tokens, validate_job, REVISION

class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(map(ord,text))
    def __call__(self,text,**kwargs):
        return {'input_ids':self.encode(text),'offset_mapping':[(i,i+1) for i in range(len(text))]}
    def apply_chat_template(self,messages,**kwargs):
        return '<system>'+messages[0]['content']+'</system><user>'+messages[1]['content']+'</user><assistant>'

case = dict(text_prefix='A semantic sentence. A literal keyword.', readout_text='Read.',
            target_span='A semantic sentence.', keyword_span='A literal keyword.',
            probe_question='Yes or no?')
ids,spans,text,audit=segmented_tokens(CharacterTokenizer(),case)
assert ''.join(map(chr,ids))==text
for scope,expected in [('carrier',case['target_span']),('keyword',case['keyword_span']),('readout_text','Read.')]:
    a,b=spans[scope]; assert text[a:b]==expected
assert spans['carrier'][1] < spans['keyword'][1] < spans['readout'][1]
pids,pspans,ptext,paudit=segmented_tokens(CharacterTokenizer(),case,probe=True)
assert 'Yes or no?</user><assistant>' in ptext and 'Read.' not in ptext
assert paudit['canonical_tokens'] and not audit['segmented_differs']
old=np.array([-3.,-1.,0.,1.,3.]); active=np.array([False,True,True,True,False])
assert np.array_equal(apply_numpy(old,{'mode':'restore'},active),old)
assert np.array_equal(apply_numpy(old,{'mode':'add','value':1.,'clip':[-2.,2.]},active),[-3.,0.,1.,2.,3.])
assert np.array_equal(apply_numpy(old,{'mode':'suppress_high','value':0.,'threshold':.5},active),[-3.,-1.,0.,0.,3.])
assert np.array_equal(apply_numpy(old,{'mode':'toward','value':1.,'fraction':.5,'direction':1},np.ones(5,dtype=bool)),[-1.,0.,.5,1.,3.])
assert np.array_equal(apply_numpy(old,{'mode':'toward','value':-1.,'fraction':.5,'direction':-1},np.ones(5,dtype=bool)),[-3.,-1.,-.5,0.,1.])
validate_job({'revision':REVISION,'conditions':[{'condition_id':'baseline'},{'condition_id':'bounded','operations':[{'ingredient':'genki','mode':'toward','value':1.,'fraction':.5,'direction':1,'scope':'readout'}]}]})
assert 'mlx.core' not in sys.modules
print('CPU measurement math/token-boundary/no-GPU-import checks passed')
