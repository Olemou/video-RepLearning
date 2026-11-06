"""Synthetic integration verifies fitting excludes selection sources."""
import json
from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_dense_spill as entry
from vision_model.models.dense_encoder import file_sha256


@pytest.mark.parametrize('cold_pixel_input',[False,True])
def test_dense_cli_prototypes_and_weights_exclude_selection_features(tmp_path,cold_pixel_input):
    torch.manual_seed(19)
    labels=torch.tensor([0,0,1,1,0,1])
    tokens=torch.randn(6,2,3,4,8)
    tokens[labels==1,:,:,0]+=3
    cache=tmp_path/'cache.pt';candidates=tmp_path/'candidates.pt'
    valid=torch.ones(6,3,dtype=torch.bool);content=torch.ones(6,2,2,dtype=torch.bool)
    data=dict(tokens=tokens,labels=labels,ids=[f'source_{i}' for i in range(6)],valid=valid,
        content=content,train_indices=torch.arange(4),validation_indices=torch.tensor([4,5]))
    target=torch.zeros(6,2,3,4,dtype=torch.long);target[labels==1,:,:,0]=1
    pseudo={k:v for k,v in data.items() if k!='tokens'};pseudo['targets']=target
    pseudo['cold_contrast']=torch.rand(6,3,4)
    torch.save(pseudo,candidates)
    results=[]
    for version in range(2):
        if version: data['tokens'][4:]*=1000
        torch.save(data,cache)
        (tmp_path/'report.json').write_text(json.dumps(dict(metadata=dict(cache_sha256=file_sha256(cache),reference_sources=data['ids'][:2],
            input_alignment='encoder_input_uint8_reconstruction_v1'),calibration=dict(high_q999=.3))))
        output=tmp_path/f'fit_{version}'
        entry.main(['--cache',str(cache),'--candidates',str(candidates),'--output',str(output),'--updates','2']+
                   (['--cold-pixel-input'] if cold_pixel_input else []))
        results.append(torch.load(output/'checkpoint.pt',weights_only=False))
        rep=torch.load(output/'representations.pt',weights_only=False)
        assert len(rep['rows'])==6 and results[-1]['metadata']['optimizer_updates']==2
        assert results[-1]['metadata']['manual_annotations']==0
        assert results[-1]['view_time_exposures'].sum()==4*2*3*2
        assert results[-1]['view_time_exposures'][4:].sum()==0
        for _,row,error in rep['rows']:assert error<2e-6
    for key,value in results[0]['model'].items():
        torch.testing.assert_close(value,results[1]['model'][key],atol=0,rtol=0)
