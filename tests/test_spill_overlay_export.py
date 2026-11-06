"""Check numerical peaks and the actual exported blue-to-red video contract."""
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image,ImageFont
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_and_export_spill_overlays import evidence_peak,render_blue_red_frame,encode_video


def test_peak_uses_content_not_padding_or_an_assumed_spill_region():
    grid=torch.zeros(8,8);grid[0,0]=100;grid[1,1]=2
    support=torch.zeros(8,8,dtype=torch.bool);support[1:7]=True
    peak=evidence_peak(grid,support,480,640)
    assert peak['native_row_col']==[1,1]
    assert peak['source_center_xy']==[120,40]
    assert peak['value']==2
    assert evidence_peak(torch.zeros(8,8),support,480,640) is None


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'),reason='FFmpeg required')
def test_encoded_blue_red_video_has_low_blue_and_shared_scale_without_color_bleed(tmp_path):
    support=torch.zeros(8,8,dtype=torch.bool);support[1:7]=True
    original=Image.fromarray(np.full((480,640,3),100,dtype=np.uint8))
    frames=[]
    for maximum in (2.,1.):
        grid=torch.zeros(8,8);grid[1,1]=maximum
        frame,peaks=render_blue_red_frame(original,{'FT0':grid,'FT1':grid},
            support=support,scale=2.,font=ImageFont.load_default(),title='test',detail='test',predictions={'FT0':1,'FT1':1})
        assert peaks['FT1']['value']==maximum
        frames.append(frame)
    path=tmp_path/'comparison.mp4'
    info=encode_video(path,frames,4)
    assert int(info['nb_read_frames'])==2
    decoded=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(path),
        '-f','rawvideo','-pix_fmt','rgb24','-threads','1','-']),dtype=np.uint8).reshape(2,392,1600,3)
    # Avoid cross markers and headers. Cell boundary at x=360 must stay sharp.
    low=decoded[0,90,355].astype(int)
    high=decoded[0,90,370].astype(int)
    half=decoded[1,90,370].astype(int)
    assert low[2]>low[0]+70
    assert high[0]>high[2]+70
    assert half[1]>half[0]+70  # 1/2 of global scale is green, not locally red.
    assert np.max(np.abs(decoded[:,90,100].astype(int)-100))<=5  # Original panel.
    assert decoded[0,90,359,2]>decoded[0,90,359,0]+60
