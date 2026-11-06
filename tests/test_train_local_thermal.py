import json,sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_local_thermal as entry
from vision_model.models.dense_encoder import file_sha256


@pytest.mark.parametrize('local_conv',[False,True])
def test_local_fitting_and_threshold_exclude_selection_features(tmp_path,local_conv):
    torch.manual_seed(39)
    cache=tmp_path/'cache.pt';pairpath=tmp_path/'pairs.pt'
    data=dict(ids=[f'source{i}' for i in range(6)],labels=torch.tensor([0,0,1,1,0,1]),
        train_indices=torch.arange(4),validation_indices=torch.tensor([4,5]),
        features=torch.randn(6,2,2,8,4,4),valid=torch.ones(6,2,dtype=torch.bool))
    r=torch.zeros(4,4);r[1,1]=1;a=torch.zeros(4,4);a[2,2]=1
    pairs=[[[dict(region=r,ring=a,confidence=.8)] for _ in range(2)] for _ in range(6)]
    results=[];reports=[]
    for version in range(2):
        if version:data['features'][4:]*=1000
        torch.save(data,cache);torch.save(dict(ids=data['ids'],pairs=pairs,cache_sha256=file_sha256(cache)),pairpath)
        out=tmp_path/f'fit{version}'
        entry.main(['--cache',str(cache),'--pairs',str(pairpath),'--output',str(out),'--updates','2']+(['--local-conv'] if local_conv else []))
        results.append(torch.load(out/'checkpoint.pt',weights_only=False));reports.append(json.loads((out/'report.json').read_text()))
        assert results[-1]['view_exposures'].sum()==4*2*2*2
    for k,v in results[0]['model'].items():torch.testing.assert_close(v,results[1]['model'][k],atol=0,rtol=0)
    assert reports[0]['calibration']==reports[1]['calibration']
