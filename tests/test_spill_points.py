import json
import copy
import pytest
import torch
from vision_model.data.spill_points import annotation_template,load_spill_points,point_cell,require_training_points


def setup_points(tmp_path):
    raw=dict(ids=['cold','clean'],video_sha256=['a'*64,'b'*64],labels=torch.tensor([1,0]),
        frame_indices=torch.tensor([[3,7],[3,7]]),valid=torch.ones(2,2,dtype=torch.bool))
    records={name:dict(height=480,width=640) for name in raw['ids']}
    payload=annotation_template(raw,records)
    row=payload['records'][0];row.update(verified=True,verified_by='test fixture reviewer',
        points=[dict(x=100,y=100,label=1),dict(x=300,y=300,label=0)])
    path=tmp_path/'points.json';path.write_text(json.dumps(payload))
    return raw,records,payload,path


def test_only_verified_cells_are_labelled_and_mirror_matches_source_coordinates(tmp_path):
    raw,records,_,path=setup_points(tmp_path)
    targets,content,coverage=load_spill_points(path,raw,records)
    assert targets.shape==(2,2,2,256)
    assert int((targets>=0).sum())==4
    assert targets[0,0,0,point_cell(100,100,480,640)]==1
    assert targets[0,1,0,point_cell(100,100,480,640,mirror=True)]==1
    assert (targets[1]==-1).all()  # Clean episode label is not silently expanded.
    assert (targets.masked_select(~content[:,None,None].expand_as(targets))==-1).all()
    assert coverage['identity_foreground_cells']==1
    assert require_training_points(targets,torch.tensor([0]))==dict(foreground=1,background=1)
    with pytest.raises(ValueError,match='verified training'):
        require_training_points(targets,torch.tensor([1]))  # Validation cannot rescue absent training labels.


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(source_sha256='c'*64),lambda r:r.update(frame_index=4),
    lambda r:r.update(verified=False),lambda r:r.update(verified_by=''),
    lambda r:r.update(width=641),lambda r:r['points'][0].update(x=-1),
    lambda r:r['points'][0].update(label=True),
    lambda r:r['points'].append(dict(x=100,y=100,label=0))])
def test_untrusted_or_ambiguous_annotations_are_rejected(tmp_path,mutation):
    raw,records,payload,path=setup_points(tmp_path)
    mutation(payload['records'][0]);path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):load_spill_points(path,raw,records)


def test_clean_foreground_and_padded_endpoint_rejected(tmp_path):
    raw,records,payload,path=setup_points(tmp_path)
    raw['labels'][0]=0
    with pytest.raises(ValueError,match='contradicts'):load_spill_points(path,raw,records)
    raw['labels'][0]=1;raw['valid'][0,0]=False
    with pytest.raises(ValueError,match='valid sampled'):load_spill_points(path,raw,records)


def test_empty_template_is_not_a_training_dataset_and_asymmetric_mapping_is_explicit(tmp_path):
    raw,records,_,path=setup_points(tmp_path)
    path.write_text(json.dumps(annotation_template(raw,records)))
    labels,_,coverage=load_spill_points(path,raw,records)
    assert (labels==-1).all() and coverage['verified_records']==0
    with pytest.raises(ValueError):require_training_points(labels,torch.arange(2))
    # 319x191 letterboxing has left/right padding 51/52: mirror source first.
    assert point_cell(0,150,319,191,mirror=True)==point_cell(190,150,319,191)
    assert point_cell(190,150,319,191,mirror=True)==point_cell(0,150,319,191)
