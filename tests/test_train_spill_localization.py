"""Small synthetic cache exercises the real CLI; it is not spill evidence."""
import json
from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_spill_localization as entry
from vision_model.data.spill_points import annotation_template
from vision_model.models.encoder import VJEPA2Encoder
from vision_model.models.checkpoint import VJEPA2Architecture
from vision_model.training.spill_localization import SpillPatchHead,SpillPrefixStudent
from vision_model.training.trajectory import TrajectoryModel


def fixture_cache(tmp_path):
    torch.manual_seed(83)
    dataset=tmp_path/'dataset';dataset.mkdir()
    names=[f'source_{i}' for i in range(6)];records={};hashes=[]
    for i,name in enumerate(names):
        source=dataset/(name+'.mp4');source.write_bytes(f'synthetic cache source {i}'.encode())
        hashes.append(entry.digest(source))
        records[name]=dict(video_id=name,video_path=source.name,width=16,height=16)
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(''.join(json.dumps(v)+'\n' for v in records.values()))
    encoder=VJEPA2Encoder(VJEPA2Architecture(frames_per_clip=6,crop_size=16,patch_size=8,tubelet_size=2,
        hidden_size=96,num_hidden_layers=2,num_attention_heads=3,mlp_ratio=2.)).eval()
    student=SpillPrefixStudent(encoder.blocks[-1],encoder.norm,head=SpillPatchHead(96))
    rows=[];means=[]
    with torch.no_grad():
        for _ in names:
            prefixes=[[torch.randn(p*4,96) for p in (1,2,3)] for view in range(2)]
            reference=torch.stack([torch.stack([student(h[None],frames=p+1,height=2,width=2,capture=True)['tokens'][0]
                for p,h in enumerate(view)]) for view in prefixes])
            rows.append(dict(prefixes=prefixes,reference=reference));means.append(reference.mean(-2))
    raw=dict(ids=names,labels=torch.tensor([0,0,1,1,0,1]),features=torch.stack(means),
        valid=torch.ones(6,3,dtype=torch.bool),timestamps=torch.arange(3).float().expand(6,-1),
        frame_indices=torch.tensor([1,3,5]).expand(6,-1),video_sha256=hashes,manifest=str(manifest),
        dataset_root=str(dataset),temporal_context_policy='past_only_prefix_last_tubelet_v1',
        checkpoint_sha256='d'*64,model_id='synthetic-test-fixture',checkpoint_image_size=16)
    cache=tmp_path/'teacher.pt';torch.save(raw,cache)
    trajectory=TrajectoryModel(96,16,8)
    ck=tmp_path/'trajectory.pt';torch.save(dict(width=96,hidden=16,embedding_dim=8,model=trajectory.state_dict(),
        train_source_ids=names[:4],validation_source_ids=names[4:],cache_sha256=entry.digest(cache)),ck)
    prefix=tmp_path/'prefix.pt';torch.save(dict(ids=names,labels=raw['labels'],valid=raw['valid'],timestamps=raw['timestamps'],
        views=['identity','mirror'],rows=rows,train_indices=torch.arange(4),validation_indices=torch.tensor([4,5]),
        suffix_state={k:v for k,v in student.state_dict().items() if not k.startswith('local_head.')},
        teacher_cache_sha256=entry.digest(cache),encoder_sha256=raw['checkpoint_sha256'],
        temporal_context_policy=raw['temporal_context_policy'],grid=(2,2),
        suffix_architecture=dict(width=96,heads=3,mlp_ratio=2.,rope_dim=16)),prefix)
    points=annotation_template(raw,records)
    for row in points['records']:
        row.update(verified=True,verified_by='synthetic test fixture')
        label=int(raw['labels'][names.index(row['source_id'])])
        row['points']=[dict(x=11,y=11,label=0)]
        if label:row['points'].append(dict(x=3,y=3,label=1))
    annotations=tmp_path/'points.json';annotations.write_text(json.dumps(points))
    return ['--teacher-cache',str(cache),'--prefix-cache',str(prefix),'--trajectory-checkpoint',str(ck),
            '--annotations',str(annotations)],ck,cache,prefix


def test_real_cli_probe_then_joint_preserves_teacher_and_changes_only_allowed_weights(tmp_path):
    args,trajectory,teacher,prefix=fixture_cache(tmp_path)
    hashes=[entry.digest(p) for p in (trajectory,teacher,prefix)]
    probe=tmp_path/'probe';joint=tmp_path/'joint'
    entry.main(args+['--stage','probe','--output',str(probe),'--updates','2','--eval-every','1'])
    probe_report=json.loads((probe/'report.json').read_text())
    assert probe_report['evaluations'][-1]['original_state_max_drift']==0
    assert probe_report['evaluations'][-1]['validation']['points']['verified_cells']>0
    entry.main(args+['--stage','joint','--probe-checkpoint',str(probe/'checkpoint.pt'),
        '--fg-weight','.2','--output',str(joint),'--updates','2','--eval-every','1'])
    saved=torch.load(joint/'checkpoint.pt',map_location='cpu',weights_only=False)
    original=torch.load(trajectory,map_location='cpu',weights_only=False)
    for name,value in original['model'].items():torch.testing.assert_close(saved['trajectory'][name],value,atol=0,rtol=0)
    initial=torch.load(prefix,map_location='cpu',weights_only=False)['suffix_state']
    assert max(float((saved['model'][name]-value).abs().max()) for name,value in initial.items() if '.attn.' in name)>0
    for name,value in initial.items():
        if '.attn.' not in name:torch.testing.assert_close(saved['model'][name],value,atol=0,rtol=0)
    assert [entry.digest(p) for p in (trajectory,teacher,prefix)]==hashes
    assert saved['metadata']['frozen_parameter_max_drift']==0
    assert saved['metadata']['optimizer_reset_before_update'] is None
    assert saved['view_time_exposures'].sum()==4*2*3*2
    with pytest.raises(FileExistsError):entry.main(args+['--output',str(probe),'--updates','1'])


def test_template_and_preflight_never_invent_training_labels(tmp_path,capsys):
    args,_,_,_=fixture_cache(tmp_path)
    blank=tmp_path/'blank.json'
    entry.main(args+['--write-template',str(blank)])
    assert all(not row['points'] and row['verified'] is False for row in json.loads(blank.read_text())['records'])
    with pytest.raises(SystemExit) as result:entry.main(args[:-2]+['--annotations',str(blank),'--validate-only'])
    assert result.value.code==2 and '"ready_for_training": false' in capsys.readouterr().out
    with pytest.raises(ValueError,match='verified training'):
        entry.main(args[:-2]+['--annotations',str(blank),'--output',str(tmp_path/'must_not_train')])
    assert not (tmp_path/'must_not_train').exists()


def test_automatic_candidates_are_separate_and_trajectory_control_freezes_local_head(tmp_path):
    args,trajectory,teacher,prefix=fixture_cache(tmp_path)
    raw=torch.load(teacher,weights_only=False);candidate=tmp_path/'automatic';candidate.mkdir()
    targets=torch.zeros(6,2,3,4,dtype=torch.long)
    targets[raw['labels']==1,:,:,0]=1
    pseudo=candidate/'maps.pt'
    torch.save(dict(ids=raw['ids'],valid=raw['valid'],timestamps=raw['timestamps'],train_indices=torch.arange(4),
        validation_indices=torch.tensor([4,5]),content=torch.ones(6,4,dtype=torch.bool),pseudo_targets={'block5_raw':targets}),pseudo)
    metadata=dict(teacher_cache_sha256=entry.digest(teacher),trajectory_checkpoint_sha256=entry.digest(trajectory),source_sha256=raw['video_sha256'])
    document=dict(metadata=metadata,cases={'block5_raw':{'normal_sources':raw['ids'][:2]}})
    (candidate/'results.json').write_text(json.dumps(document))
    auto=args[:-2]+['--pseudo-cache',str(pseudo)]
    probe=tmp_path/'auto_probe';joint=tmp_path/'auto_joint';control=tmp_path/'auto_control'
    entry.main(auto+['--output',str(probe),'--updates','1'])
    saved=torch.load(probe/'checkpoint.pt',weights_only=False)
    assert saved['metadata']['supervision_kind']=='automatic_unverified' and saved['metadata']['annotation_sha256'] is None
    report=json.loads((probe/'report.json').read_text())
    assert 'points' not in report['evaluations'][-1]['validation']
    assert 'verified_cells' not in report['evaluations'][-1]['validation']['pseudo_candidate_agreement']
    for path,extra in ((control,['--trajectory-only']),(joint,[])):
        entry.main(auto+['--stage','joint','--probe-checkpoint',str(probe/'checkpoint.pt'),'--fg-weight','.2',
            '--output',str(path),'--updates','1']+extra)
    controlled=torch.load(control/'checkpoint.pt',weights_only=False)
    learned=torch.load(joint/'checkpoint.pt',weights_only=False)
    assert all(torch.equal(v,controlled['local_head'][k]) for k,v in saved['local_head'].items())
    assert any(not torch.equal(v,learned['local_head'][k]) for k,v in saved['local_head'].items())
    document['cases']['block5_raw']['normal_sources'].append(raw['ids'][4])
    (candidate/'results.json').write_text(json.dumps(document))
    with pytest.raises(ValueError,match='exactly training clean'):entry.main(auto+['--validate-only'])
    document['cases']['block5_raw']['normal_sources']=raw['ids'][:2]
    (candidate/'results.json').write_text(json.dumps(document))
    stored=torch.load(pseudo,weights_only=False);stored['pseudo_targets']['block5_raw'][0,0,0,0]=1;torch.save(stored,pseudo)
    with pytest.raises(ValueError,match='clean foreground'):entry.main(auto+['--validate-only'])
    stored['pseudo_targets']['block5_raw'].zero_();torch.save(stored,pseudo)
    with pytest.raises(ValueError,match='Within-spill background coverage failed'):
        entry.main(auto+['--output',str(tmp_path/'no_candidates'),'--updates','1'])
    assert not (tmp_path/'no_candidates').exists()


def _entry_worker(rank,port,args):
    import os
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2')
    entry.main(args)


def test_two_rank_cli_runs_probe_and_joint_with_unique_evaluation_sources(tmp_path):
    import socket
    import torch.multiprocessing as mp
    args,_,_,_=fixture_cache(tmp_path)
    def launch(arguments):
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        mp.spawn(_entry_worker,args=(port,arguments),nprocs=2)
    probe=tmp_path/'ddp_probe';joint=tmp_path/'ddp_joint'
    launch(args+['--output',str(probe),'--updates','1'])
    launch(args+['--stage','joint','--fg-weight','.2','--probe-checkpoint',str(probe/'checkpoint.pt'),
        '--output',str(joint),'--updates','1'])
    report=json.loads((joint/'report.json').read_text())
    assert report['metadata']['rank_eval_source_counts']==[3,3]
    saved=torch.load(joint/'representations.pt',weights_only=False)
    assert len(saved['ids'])==len(set(saved['ids']))==6
