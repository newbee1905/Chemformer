import torch; 
p='models/fined-tuned/uspto_50/last.ckpt' 
m=torch.load(p, map_location='cpu')
hp=m.get('hyper_parameters',{})
if 'vocabulary_size' not in hp and 'vocab_size' in hp: 
    hp['vocabulary_size']=hp.pop('vocab_size')
elif 'vocabulary_size' not in hp and 'vocab_size' not in hp: 
    hp['vocabulary_size']=523
m['hyper_parameters']=hp
out='models/fined-tuned/uspto_50/last_v2.ckpt'
torch.save(m, out)
print('Wrote', out)