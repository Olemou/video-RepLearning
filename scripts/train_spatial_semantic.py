#!/usr/bin/env python3
"""E0/E1/E2: frozen temporal control and decoder-free spatial mirror ablation.

Uses genuine paired causal patch caches, global ViCo Eq.10 and proposed TV
equivariance. No pixel supervision, auxiliary classification loss or decoder.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from vision_model.training.spatial_semantic import (SpatialSemanticBranch,
    spatial_objective,mirror_total_variation,attention_diagnostics,SpatialPairedForward)
from vision_model.training.trajectory import TrajectoryModel,trajectory_loss_from_output
from vision_model.training.vico import vico_pairwise_contrastive
from vision_model.training.distributed import initialize_distributed,audit_distributed_indices
from diagnose_spill_representation import digest,recognition

EXPERIMENT=ROOT/"reports/experiments/trajectory_80_sources_real_ddp_all_losses_20261021"
DEFAULT_SPATIAL=EXPERIMENT/"research_diagnosis/spatial_matched_precision/spatial_tokens.pt"
DEFAULT_TEACHER=ROOT/"reports/experiments/trajectory_80_sources_12f_lrsearch_20261021/causal_features.pt"


def content_masks(manifest,ids,grid):
    """Retain content patches with >50% observed pixels; no spill ROI."""
    records={r["video_id"]:r for r in (json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip())}
    h,w=grid
    masks=[]
    for name in ids:
        r=records[name]
        scale=min(256/r["height"],256/r["width"])
        rh,rw=round(r["height"]*scale),round(r["width"]*scale)
        top,left=(256-rh)//2,(256-rw)//2
        canvas=torch.zeros(256,256)
        canvas[top:top+rh,left:left+rw]=1
        observed=F.avg_pool2d(canvas[None,None],(256//h,256//w))[0,0]>.5
        # Native grid flip must match a fresh full-frame horizontal reflection.
        if left!=256-left-rw or not torch.equal(observed,observed.flip(-1)):
            raise ValueError(f"Asymmetric letterbox requires explicit coordinate mapping: {name}")
        masks.append(observed)
    return torch.stack(masks)


def read_inputs(a):
    c=torch.load(a.spatial_cache,map_location="cpu",weights_only=False,mmap=True)
    raw=torch.load(a.teacher_cache,map_location="cpu",weights_only=False)
    ck=torch.load(a.temporal_checkpoint,map_location="cpu",weights_only=False)
    if c["temporal_context_policy"]!="past_only_prefix_last_tubelet_v1" or c["cache_feature_max_error"]>.08:
        raise ValueError("Unverified causal/precision cache")
    if c["ids"]!=raw["ids"] or not torch.equal(c["labels"],raw["labels"]) or not torch.equal(c["valid"],raw["valid"]):
        raise ValueError("Teacher/spatial cache alignment mismatch")
    if not torch.equal(c["timestamps"],raw["timestamps"]) or c["views"]!=["identity","mirror"]:
        raise ValueError("Timestamp or paired-view mismatch")
    if len(set(raw["video_sha256"]))!=len(c["ids"]):
        raise ValueError("Duplicate source hashes")
    if digest(a.temporal_checkpoint)!=c["checkpoint_sha256"] or digest(a.teacher_cache)!=c["cache_sha256"]:
        raise ValueError("Temporal checkpoint/cache hash mismatch")
    if ck["cache_sha256"]!=c["cache_sha256"] or raw["checkpoint_sha256"]!=c["encoder_sha256"] or raw["normalization"]!=c["normalization"]:
        raise ValueError("Encoder or normalization mismatch")
    train=torch.tensor([c["ids"].index(i) for i in ck["train_source_ids"]])
    val=torch.tensor([c["ids"].index(i) for i in ck["validation_source_ids"]])
    if sorted(train.tolist()+val.tolist())!=list(range(len(c["ids"]))):
        raise ValueError("Invalid source split")
    if not torch.equal(train,c["train_indices"]) or not torch.equal(val,c["validation_indices"]):
        raise ValueError("Spatial cache split mismatch")
    return c,raw,ck,train,val


def flatten_pair(bank,valid,content):
    n,views,t,h,w,d=bank.shape
    if views!=2:raise ValueError("Need identity and mirrored views")
    return (bank.reshape(n*2,t,h,w,d).float(),
            valid[:,None].expand(-1,2,-1).reshape(n*2,t),
            content[:,None].expand(-1,2,-1,-1).reshape(n*2,h,w))


def distributed_gradient_probe(model,bank,valid,content,labels,ids,epoch,config):
    """Probe independent parameter copies; never bypass DDP accumulation hooks."""
    probe=SpatialSemanticBranch(model.class_prototypes,**config).to(bank.device).eval()
    probe.load_state_dict(model.state_dict(),strict=True)
    out=probe(*flatten_pair(bank,valid,content))
    _,detail=spatial_objective(out,labels,ids,valid,content,epoch=epoch)
    parameters=tuple(probe.parameters())
    vectors=[]
    for key in ("semantic_loss","mirror_loss"):
        grads=torch.autograd.grad(detail[key],parameters,retain_graph=True,allow_unused=True)
        vector=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(parameters,grads)])
        if dist.is_initialized():
            dist.all_reduce(vector)
            vector/=dist.get_world_size()
        vectors.append(vector)
    s,m=vectors
    shared=torch.cat([torch.full((p.numel(),),not name.startswith("semantic."),dtype=torch.bool,device=s.device)
                      for name,p in probe.named_parameters()])
    ss,mm=s[shared],m[shared]
    return dict(semantic_gradient_norm=float(s.norm()),mirror_gradient_norm=float(m.norm()),
                shared_semantic_gradient_norm=float(ss.norm()),shared_mirror_gradient_norm=float(mm.norm()),
                gradient_cosine=float((ss@mm)/(ss.norm()*mm.norm()).clamp_min(1e-30)),
                semantic_loss=float(detail["semantic_loss"].detach()),mirror_loss=float(detail["mirror_loss"].detach()),
                temporal_gradient_to_spatial=0.)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spatial-cache",type=Path,default=DEFAULT_SPATIAL)
    p.add_argument("--teacher-cache",type=Path,default=DEFAULT_TEACHER)
    p.add_argument("--temporal-checkpoint",type=Path,default=EXPERIMENT/"training/trial_01_checkpoint.pt")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--updates",type=int,default=100)
    p.add_argument("--seed",type=int,default=239)
    p.add_argument("--learning-rate",type=float,default=1e-3)
    p.add_argument("--microbatch-sources",type=int,default=4)
    p.add_argument("--eval-every",type=int,default=10)
    p.add_argument("--mirror-weight",type=float,help="Otherwise calibrate once from training gradients")
    p.add_argument("--mirror-policy",choices=["guarded","warmup"],default="guarded")
    p.add_argument("--reuse-baseline",type=Path,help="Existing matched E1_no_mirror directory; train E2 only")
    p.add_argument("--layer",type=int,default=23)
    a=p.parse_args()
    if a.updates<1 or a.microbatch_sources<1 or a.eval_every<1:raise ValueError("Invalid budget")
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    ctx=initialize_distributed()
    c,raw,ck,train,val=read_inputs(a)
    if ctx.is_main:a.output.mkdir(parents=True,exist_ok=False)
    if ctx.world_size>1:dist.barrier()
    assigned=torch.arange(ctx.rank,len(c["ids"]),ctx.world_size)
    _,rank_counts=audit_distributed_indices(assigned.tolist(),len(c["ids"]))
    slot=c["layers"].index(a.layer)
    spatial=c["tokens"][assigned,:,:,slot]
    side=int(spatial.shape[-2]**.5)
    if side*side!=spatial.shape[-2]:raise ValueError("Non-square native patch grid")
    bank=spatial.reshape(*spatial.shape[:3],side,side,spatial.shape[-1]).to(ctx.device)
    content=content_masks(raw["manifest"],c["ids"],(side,side))
    local_content=content[assigned].to(ctx.device)
    local_valid=c["valid"][assigned].to(ctx.device)
    lookup={int(v):i for i,v in enumerate(assigned)}
    local_train=torch.tensor([lookup[int(v)] for v in train if int(v) in lookup],device=ctx.device)
    train_ids=assigned[local_train.cpu()].to(ctx.device)
    all_train=[None]*ctx.world_size
    if ctx.world_size>1:dist.all_gather_object(all_train,train_ids.cpu().tolist())
    else:all_train=[train_ids.cpu().tolist()]
    if sorted(v for shard in all_train for v in shard)!=sorted(train.tolist()):raise RuntimeError("Distributed training ownership mismatch")
    train_counts=[len(v) for v in all_train]
    if min(train_counts)==0:raise ValueError("Every rank needs training sources")
    teacher=TrajectoryModel(ck["width"],ck["hidden"],ck["embedding_dim"]).to(ctx.device).eval().requires_grad_(False)
    teacher.load_state_dict(ck["model"],strict=True)
    tx=raw["features"][assigned].to(ctx.device)
    tv=raw["valid"][assigned].to(ctx.device)
    tt=raw["timestamps"][assigned].to(ctx.device)
    with torch.no_grad():
        baseline_local=teacher(tx.flatten(0,1),tv[:,None].expand(-1,2,-1).flatten(0,1),tt[:,None].expand(-1,2,-1).flatten(0,1))
    @torch.no_grad()
    def future_outputs(output):
        stamps=tt[:,None].expand(-1,2,-1).flatten(0,1)
        state=F.layer_norm(output["states"],(ck["width"],))
        result={}
        for horizon in (1,2,4):
            if horizon>=stamps.shape[1]:continue
            delta=(stamps[:,horizon:]-stamps[:,:-horizon])[...,None]
            result[horizon]=state[:,:-horizon]+teacher.future_delta(torch.cat((output["dynamics"][:,:-horizon],delta),-1))
        return result
    baseline_future=future_outputs(baseline_local)
    gathered=[None]*ctx.world_size if ctx.is_main else None
    payload=(assigned.tolist(),{k:v.cpu() for k,v in baseline_local.items()})
    if ctx.world_size>1:dist.gather_object(payload,gathered,dst=0)
    else:gathered=[payload]
    baseline=None
    if ctx.is_main:
        combined={k:torch.zeros(len(c["ids"]),2,*v.shape[1:]) for k,v in baseline_local.items()}
        for indices,parts in gathered:
            for key,value in parts.items():combined[key][indices]=value.reshape(len(indices),2,*value.shape[1:])
        z=combined["embedding"][:,0]
        prototypes=torch.stack([F.normalize(z[train][c["labels"][train]==i].mean(0),dim=0) for i in (0,1)])
        baseline=dict(accuracy=recognition(z,c["labels"],train,val),temporal={})
        for split,indices in (("train",train),("validation",val)):
            output={k:v[indices].flatten(0,1).to(ctx.device) for k,v in combined.items()}
            with torch.no_grad():
                _,detail,_=trajectory_loss_from_output(teacher,raw["features"][indices].to(ctx.device),
                    c["valid"][indices].to(ctx.device),c["timestamps"][indices].to(ctx.device),c["labels"][indices].to(ctx.device),output,epoch=20,curriculum_epochs=20)
            baseline["temporal"][split]={k:float(v) for k,v in detail.items()}
        (a.output/"E0_baseline.json").write_text(json.dumps(baseline,indent=2)+"\n")
    else:prototypes=torch.zeros(2,ck["embedding_dim"])
    prototypes=prototypes.to(ctx.device)
    if ctx.world_size>1:dist.broadcast(prototypes,src=0)
    config=dict(width=ck["width"],hidden=128,embedding_dim=128,heads=4,layers=1)
    torch.manual_seed(a.seed)
    initial=SpatialSemanticBranch(prototypes,**config).to(ctx.device)
    initial_state=copy.deepcopy(initial.state_dict())
    train_bank=bank[local_train]
    train_valid=local_valid[local_train]
    train_content=local_content[local_train]
    train_labels=c["labels"][train_ids.cpu()].to(ctx.device)
    calibration=distributed_gradient_probe(initial,train_bank,train_valid,train_content,train_labels,train_ids,0,config)
    selected_weight=(a.mirror_weight if a.mirror_weight is not None else
        min(1.,.1*calibration["shared_semantic_gradient_norm"]/max(calibration["shared_mirror_gradient_norm"],1e-12)))
    if selected_weight<=0:raise ValueError("Nonzero comparison needs a positive mirror weight")
    calibration.update(selected_mirror_weight=selected_weight,
        selection="Explicit setting" if a.mirror_weight is not None else "Training-only initialization gradients on shared spatial parameters; cap at 1 and target mirror gradient <=10% of ViCo",
        subsequent_guard=("At ten-update probes, only reduce weight if shared mirror gradient exceeds 25% of shared semantic gradient" if a.mirror_policy=="guarded" else "Fixed calibrated target with 20-update linear warmup; no relative-gradient decay after semantic convergence"),validation_used=False)
    if ctx.is_main:(a.output/"gradient_calibration.json").write_text(json.dumps(calibration,indent=2)+"\n")
    del initial

    @torch.no_grad()
    def evaluate(model,step):
        model.eval()
        rows=[]
        for start in range(0,len(assigned),a.microbatch_sources):
            stop=min(start+a.microbatch_sources,len(assigned))
            out=model(*flatten_pair(bank[start:stop],local_valid[start:stop],local_content[start:stop]))
            for j,index in enumerate(assigned[start:stop]):
                sl=slice(2*j,2*j+2)
                rows.append(dict(index=int(index),embedding=out["embedding"][sl].cpu(),attention=out["attention"][sl].cpu(),head_attention=out["head_attention"][sl].cpu()))
        gathered=[None]*ctx.world_size if ctx.is_main else None
        if ctx.world_size>1:dist.gather_object(rows,gathered,dst=0)
        else:gathered=[rows]
        if not ctx.is_main:return None,None
        rows=sorted([r for shard in gathered for r in shard],key=lambda r:r["index"])
        if [r["index"] for r in rows]!=list(range(len(c["ids"]))):raise RuntimeError("Repeated or missing evaluation source")
        emb=torch.stack([r["embedding"] for r in rows])
        maps=torch.stack([r["attention"] for r in rows])
        result=dict(update=step,train=recognition(emb[:,0],c["labels"],train,train,exclude_self=True),
                    validation=recognition(emb[:,0],c["labels"],train,val))
        for split,indices in (("train",train),("validation",val)):
            _,detail=vico_pairwise_contrastive(emb[indices].flatten(0,1),c["labels"][indices].repeat_interleave(2),
                source_ids=indices.repeat_interleave(2),epoch=20,curriculum_epochs=20)
            diag=attention_diagnostics(maps[indices],c["valid"][indices],content[indices])
            diag={k:float(v.mean()) for k,v in diag.items()}
            tv_value=mirror_total_variation(maps[indices],c["valid"][indices],content[indices])
            result[split].update(fixed_semantic_loss=float(detail["vico_loss"]),mirror_TV=float(tv_value),
                **{k:float(v) for k,v in detail.items()},**diag)
            # Explicitly constant temporal terms are monitoring values only.
            temporal=baseline["temporal"][split]
            constant=.1*temporal["alignment_loss"]+temporal["prediction_loss"]+.1*temporal["retention_loss"]
            result[split]["monitored_total"]=float(detail["vico_loss"])+active_weight*float(tv_value)+constant
        return result,rows

    cases=[]
    conditions=[("E1_no_mirror",0.),("E2_mirror",selected_weight)]
    if a.reuse_baseline:
        saved=torch.load(a.reuse_baseline/"checkpoint.pt",map_location="cpu",weights_only=False)
        old=saved["metadata"]
        for key,value in (("seed",a.seed),("optimizer_updates",a.updates),("architecture",config),("train_source_ids",ck["train_source_ids"]),("validation_source_ids",ck["validation_source_ids"])):
            if old[key]!=value:raise ValueError(f"Baseline mismatch: {key}")
        for key,value in (("learning_rate",a.learning_rate),("layer",a.layer)):
            if saved["configuration"][key]!=value:raise ValueError(f"Baseline setting mismatch: {key}")
        torch.testing.assert_close(saved["class_prototypes"],prototypes.cpu(),atol=0,rtol=0)
        if saved["cache_provenance"]["cache_sha256"]!=c["cache_sha256"]:raise ValueError("Baseline cache mismatch")
        if ctx.is_main:
            baseline_metrics=json.loads((a.reuse_baseline/"metrics.json").read_text())["final"]
            cases.append(dict(**old,metrics=baseline_metrics,artifact_directory=str(a.reuse_baseline.resolve()),reused_matched_baseline=True))
        conditions=[("E2_mirror",selected_weight)]
    for name,active_weight in conditions:
        initial_weight=active_weight
        case_dir=a.output/name
        if ctx.is_main:case_dir.mkdir()
        torch.manual_seed(a.seed)
        model=SpatialSemanticBranch(prototypes,**config).to(ctx.device)
        model.load_state_dict(initial_state,strict=True)
        forward=SpatialPairedForward(model,a.microbatch_sources)
        wrapped=DDP(forward,device_ids=[ctx.local_rank] if ctx.device.type=="cuda" else None,broadcast_buffers=False) if ctx.world_size>1 else forward
        optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,eps=1e-12,weight_decay=0.)
        history,eval_history,gradient_history=[],[],[]
        first,_=evaluate(model,0)
        if ctx.is_main:eval_history.append(first)
        started=time.monotonic()
        for update in range(1,a.updates+1):
            if a.mirror_policy=="warmup":active_weight=initial_weight*min(update/20.,1.)
            if update==21:optimizer.state.clear()
            wrapped.train()
            optimizer.zero_grad(set_to_none=True)
            output=wrapped(train_bank,train_valid,train_content)
            loss,detail=spatial_objective(output,train_labels,train_ids,train_valid,train_content,
                                          epoch=update,mirror_weight=active_weight)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite objective")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            values=torch.tensor([float(detail["semantic_loss"].detach()),float(detail["mirror_loss"].detach()),float(norm)],device=ctx.device)
            # Full-batch rank counts can differ: weight auxiliary sample means.
            values[1]*=len(train_bank)
            if ctx.world_size>1:
                dist.all_reduce(values)
                values[0]/=ctx.world_size
                values[2]/=ctx.world_size
            values[1]/=len(train)
            row=dict(update=update,semantic=float(values[0]),mirror_TV=float(values[1]),mirror_weight=active_weight,
                     optimized_loss=float(values[0]+active_weight*values[1]),gradient_norm=float(values[2]),learning_rate=a.learning_rate)
            history.append(row)
            if update in {1,21,a.updates} or update%10==0:
                g=distributed_gradient_probe(model,train_bank,train_valid,train_content,train_labels,train_ids,update,config)
                g.update(update=update,weighted_mirror_gradient_norm=active_weight*g["mirror_gradient_norm"],mirror_weight=active_weight)
                shared_limit=.25*g["shared_semantic_gradient_norm"]/max(g["shared_mirror_gradient_norm"],1e-12)
                if a.mirror_policy=="guarded" and active_weight>shared_limit:active_weight=shared_limit
                g["next_mirror_weight"]=active_weight
                gradient_history.append(g)
            if update%a.eval_every==0 or update==a.updates:
                metrics,rows=evaluate(model,update)
                if ctx.is_main:
                    eval_history.append(metrics)
                    print(json.dumps(dict(case=name,update=update,train_ViCo=metrics["train"]["fixed_semantic_loss"],validation_accuracy=metrics["validation"]["accuracy"],validation_mirror_TV=metrics["validation"]["mirror_TV"])),flush=True)
        with torch.no_grad():
            repeated=teacher(tx.flatten(0,1),tv[:,None].expand(-1,2,-1).flatten(0,1),tt[:,None].expand(-1,2,-1).flatten(0,1))
        drift=max(float((repeated[k]-baseline_local[k].to(ctx.device)).abs().max()) for k in repeated)
        final_future=future_outputs(repeated)
        future_drift=max(float((final_future[k]-baseline_future[k]).abs().max()) for k in final_future)
        parameter_drift=max(float((v.cpu()-ck["model"][k]).abs().max()) for k,v in teacher.state_dict().items())
        drift_tensor=torch.tensor([drift,parameter_drift,future_drift],device=ctx.device)
        if ctx.world_size>1:dist.all_reduce(drift_tensor,op=dist.ReduceOp.MAX)
        if drift_tensor.any():raise RuntimeError("Frozen temporal baseline changed")
        exposures=torch.zeros(len(c["ids"]),2,dtype=torch.long,device=ctx.device)
        exposures[train_ids]=a.updates
        if ctx.world_size>1:dist.all_reduce(exposures)
        if ctx.is_main:
            metadata=dict(case=name,initial_mirror_weight=initial_weight,final_mirror_weight=active_weight,mirror_policy=a.mirror_policy,seed=a.seed,optimizer_updates=a.updates,
                optimizer_state_reset_before_update=21 if a.updates>=21 else None,
                temporal_output_max_drift=float(drift_tensor[0]),temporal_parameter_max_drift=float(drift_tensor[1]),
                future_prediction_max_drift=float(drift_tensor[2]),
                architecture=config,train_source_ids=ck["train_source_ids"],validation_source_ids=ck["validation_source_ids"],
                prototypes_from="Frozen temporal checkpoint, training identity episodes only; both class queries used for every input",elapsed_seconds=time.monotonic()-started,
                optimized_objective="ViCo(spatial embedding) + mirror_weight * TV",monitored_temporal_losses="Frozen constants, not spatial gradients",
                patch_label_supervision=False,decoder=False,encoder_frozen=True,temporal_frozen=True,
                localization_validated=False,world_size=ctx.world_size,training_rank_source_counts=train_counts,evaluation_rank_source_counts=rank_counts)
            torch.save(dict(format="spatial_semantic_v1",model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                metadata=metadata,history=history,evaluation_history=eval_history,gradient_history=gradient_history,
                view_exposures=exposures.cpu(),configuration=vars(a),class_prototypes=prototypes.cpu(),
                cache_provenance={k:c[k] for k in ("checkpoint_sha256","cache_sha256","encoder_sha256","temporal_context_policy","normalization","views")}),case_dir/"checkpoint.pt")
            torch.save(dict(ids=c["ids"],labels=c["labels"],valid=c["valid"],timestamps=c["timestamps"],content=content,rows=rows),case_dir/"representations_and_maps.pt")
            (case_dir/"metrics.json").write_text(json.dumps(dict(metadata=metadata,final=eval_history[-1],gradient_history=gradient_history),indent=2)+"\n")
            with (case_dir/"history.csv").open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
            (case_dir/"evaluation_history.json").write_text(json.dumps(eval_history,indent=2)+"\n")
            cases.append(dict(**metadata,metrics=eval_history[-1]))
        if ctx.world_size>1:dist.barrier()
        del wrapped,model,optimizer
    if ctx.is_main:
        # Checkpoints/maps already saved before any optional report rendering.
        source_files=[Path(__file__),ROOT/"src/vision_model/training/spatial_semantic.py",ROOT/"info.md",ROOT/"src/vision_model/training/vico.py",ROOT/"src/vision_model/training/trajectory.py"]
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in source_files}
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        report=dict(protocol="Fixed 64/16 source-selection split; frozen temporal branch; no masks or independent scene test",settings=settings,
                    baseline=baseline,calibration=calibration,cases=cases,source_hashes=source_hashes,
                    git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    torch_version=torch.__version__,gpu_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    optional_joint_stage="Not run: requires evidence justifying temporal adaptation; E1/E2 leave it unchanged")
        (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(dict(output=str(a.output),mirror_weight=selected_weight,temporal_drift=0.)),flush=True)
    if ctx.world_size>1:dist.destroy_process_group()


if __name__=="__main__":main()
