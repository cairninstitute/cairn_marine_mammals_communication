#!/usr/bin/env python3
"""Measure MMC residual branches, routing, per-head, and per-expert activations."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from src.model.checkpoints import load_model_from_checkpoint,architecture_label
def rms(x): return x.detach().float().square().mean().sqrt().item()
def heads(x,h): return x.detach().float().reshape(*x.shape[:-1],h,-1).square().mean((0,1,3)).sqrt().tolist()
def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--json-out',type=Path,required=True);p.add_argument('--sequence-length',type=int,default=1024);p.add_argument('--seed',type=int,default=0);p.add_argument('--tokens',type=Path);a=p.parse_args()
 ckpt,cfg,model=load_model_from_checkpoint(a.checkpoint,'cpu');model.eval()
 if a.tokens:
  import numpy as np
  x=np.load(a.tokens)
  if x.ndim==2:x=(x+np.arange(x.shape[0],dtype=x.dtype).reshape(-1,1)*1024).T.reshape(-1)
  if x.size <= a.sequence_length: raise ValueError('token file needs one token beyond --sequence-length')
  next_target=int(x[a.sequence_length]); ids=torch.from_numpy(x[:a.sequence_length]).long().unsqueeze(0);source=str(a.tokens)
 else:
  g=torch.Generator().manual_seed(a.seed);t=torch.arange(a.sequence_length);ids=(torch.randint(1,1025,(a.sequence_length,),generator=g)+(t%9)*1024).unsqueeze(0);next_target=None;source='synthetic DAC-style interleaved token IDs'
 rec=[{} for _ in model.blocks];final={};hs=[]
 for i,b in enumerate(model.blocks):
  def save(m,args,i=i):rec[i]['_x']=args[0]
  def bpre(m,args,i=i):rec[i]['residual_in_rms']=rms(args[0])
  def aout(m,args,out,i=i):y=out[0];rec[i]['attention_out_rms']=rms(y);rec[i]['post_attention_residual_rms']=rms(rec[i]['_x']+y)
  def ffout(m,args,out,i=i):rec[i]['moe_out_rms']=rms(out)
  def gate(m,args,out,i=i):
   z=out.detach().float();q=z.softmax(-1);top=z.topk(m._k,-1).indices;rec[i]['router_entropy']=(-(q*q.clamp_min(1e-12).log()).sum(-1).mean().item());rec[i]['expert_selection_fraction']=torch.bincount(top.reshape(-1),minlength=m.out_features).div(top.numel()).tolist()
  def norm(name):return lambda m,args,out,i=i:rec[i].update({name:rms(out)})
  hs += [b.register_forward_pre_hook(save),b.register_forward_pre_hook(bpre),b.attn.register_forward_hook(aout),b.ff.register_forward_hook(ffout),b.attn_norm.register_forward_hook(norm('attention_norm_out_rms')),b.ff_norm.register_forward_hook(norm('ffn_norm_out_rms'))]
  b.ff.gate._k=b.ff.top_k;hs.append(b.ff.gate.register_forward_hook(gate))
  if hasattr(b.attn,'qkv_proj'):
   def qkv(m,args,out,i=i):
    z=out.detach().float().reshape(out.shape[0],out.shape[1],3,cfg.n_heads,-1).square().mean((0,1,4)).sqrt();rec[i]['attention_heads']={'q':z[0].tolist(),'k':z[1].tolist(),'v':z[2].tolist()}
   hs.append(b.attn.qkv_proj.register_forward_hook(qkv))
  else:
   for name in ('q','k','v'):
    hs.append(getattr(b.attn,name+'_proj').register_forward_hook(lambda m,args,out,i=i,name=name:rec[i].setdefault('attention_heads',{}).update({name:heads(out,cfg.n_heads)})))
  hs.append(b.attn.out_proj.register_forward_pre_hook(lambda m,args,i=i:rec[i].setdefault('attention_heads',{}).update({'output':heads(args[0],cfg.n_heads)})))
  for e,expert in enumerate(b.ff.experts): hs.append(expert.register_forward_hook(lambda m,args,out,i=i,e=e:rec[i].setdefault('expert_output_rms',{}).update({str(e):rms(out)})))
 def logits_hook(m,args,out):
  z=out.detach().float(); probs=z.softmax(-1); entropy=-(probs*z.log_softmax(-1)).sum(-1)
  last_probs=probs[:, -1, :]; last_top=last_probs.argmax(-1)
  final.update({'logit_rms':rms(z), 'output_entropy':entropy.mean().item(), 'top_token_confidence':probs.amax(-1).mean().item(), 'next_token_logit_rms':rms(z[:, -1, :]), 'next_token_entropy':entropy[:, -1].mean().item(), 'next_token_confidence':last_probs.amax(-1).mean().item(), 'next_token_id':int(last_top[0].item())})
  if next_target is not None:
   final.update({'next_target_token_id':next_target, 'next_target_probability':float(last_probs[0,next_target]), 'next_target_rank':int((z[0,-1,:] > z[0,-1,next_target]).sum().item()+1), 'next_target_codebook':a.sequence_length % 9})
 hs += [model.norm.register_forward_hook(lambda m,args,out: final.update({'final_hidden_rms':rms(out)})), model.lm_head.register_forward_hook(logits_hook)]
 with torch.inference_mode():model(ids)
 for h in hs:h.remove()
 for r in rec:r.pop('_x',None);r['attention_update_ratio']=r['attention_out_rms']/r['residual_in_rms'];r['moe_update_ratio']=r['moe_out_rms']/r['post_attention_residual_rms']
 report={'checkpoint':a.checkpoint,'step':ckpt.get('step'),'architecture':architecture_label(cfg),'input_source':source,'sequence_length':ids.shape[1],'layers':rec,'model_output':final};a.json_out.parent.mkdir(parents=True,exist_ok=True);a.json_out.write_text(json.dumps(report,indent=2)+'\n');print('wrote',a.json_out)
if __name__=='__main__':main()
