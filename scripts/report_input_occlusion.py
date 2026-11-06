#!/usr/bin/env python3
"""Render unnormalized signed score changes with one global comparison scale."""
import argparse
import html
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from vision_model.interpretability.map_geometry import inverse_letterbox_map,spatial_evidence_overlay
from train_spatial_semantic import DEFAULT_TEACHER
from visualize_trajectory_video import decode_selected_frames


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output-name',default='input_occlusion_figures_v2')
    a=p.parse_args();torch.set_num_threads(4)
    audit=json.loads((a.experiment/'dense_input_occlusion.json').read_text())
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    output=a.experiment/a.output_name;output.mkdir(exist_ok=False)
    scale=max(abs(v) for row in audit['rows'] for case in row['cases'].values() for rr in case['delta_margin'] for v in rr)
    try:font=ImageFont.truetype('DejaVuSans.ttf',13)
    except OSError:font=ImageFont.load_default()
    summary=[];cold=[]
    for row in audit['rows']:
        i=row['index'];r=records[row['id']];index=int(raw['frame_indices'][i][raw['valid'][i]][-1])
        frame=decode_selected_frames(Path(raw['dataset_root'])/r['video_path'],[index],frame_count=r['num_frames'],width=r['width'],height=r['height'])[index]
        canvas=Image.new('RGB',(5*256,270),'white');draw=ImageDraw.Draw(canvas)
        draw.text((5,3),f'{row["id"]} | SOURCE VALIDATION | {"cold spill" if row["label"] else "clean"} | signed score scale +/-{scale:.5f}',fill='black',font=font)
        draw.text((5,22),'Clip-wide input removals; shown over last sampled frame. These are score changes, not frame-specific spill masks.',fill='black',font=font)
        display=Image.fromarray(frame).resize((256,192),Image.Resampling.BILINEAR)
        canvas.paste(display,(0,65));draw.text((5,45),'Original',fill='black',font=font)
        result=dict(id=row['id'],label=row['label'],cases={})
        for col,(name,case) in enumerate(row['cases'].items(),1):
            grid=torch.tensor(case['delta_margin']);h,w=frame.shape[:2]
            x=np.clip(inverse_letterbox_map(grid,h,w,interpolation='nearest').numpy()/max(scale,1e-30),-1,1)
            rgb=np.ones((*x.shape,3),dtype=np.float32)
            rgb[...,0]-=np.maximum(-x,0);rgb[...,1]-=np.abs(x);rgb[...,2]-=np.maximum(x,0)
            heat=torch.from_numpy(np.round(rgb*255).astype(np.uint8))
            overlay=spatial_evidence_overlay(torch.from_numpy(np.asarray(display).copy()),grid,scale=scale,scope='input_intervention',alpha=.4)
            canvas.paste(Image.fromarray(overlay.numpy()),(col*256,65))
            draw.text((col*256+4,45),name+' input score change',fill='black',font=font)
            if col==3:
                canvas.paste(Image.fromarray(heat.numpy()).resize((256,192),Image.Resampling.NEAREST),(4*256,65));draw.text((4*256+4,45),'FT1_1em4 signed native map',fill='black',font=font)
            content=grid[1:7]
            peak=int(content.argmax());rr,cc=divmod(peak,8);rr+=1
            result['cases'][name]=dict(max_spill_score_drop=float(content.max()),min_spill_score_drop=float(content.min()),
                maximum_cell_native_row_col=[rr,cc],maximum_cell_source_center_xy=[(cc+.5)*32/256*w,((rr+.5)*32-32)/192*h],
                counterfactual_accuracy=float(np.mean(np.array(case['prediction'])==row['label'])),
                fresh_batch4_to_cache_error=case['batch4_original_to_cached_margin_max_error'])
        draw.text((5,257),'Red/blue: signed score change. Zero = transparent; opacity = 0.4*abs(delta)/shared scale. Native cells; no blur.',fill='black')
        canvas.save(output/(row['id']+'.png'));summary.append(result)
        if row['label']:cold.append(canvas)
    contact=Image.new('RGB',(cold[0].width,len(cold)*cold[0].height),'white')
    for j,im in enumerate(cold):contact.paste(im,(0,j*im.height))
    contact.save(output/'all_8_cold.png')
    body='<html><body><h1>Input-region intervention: all 16 source-validation clips</h1><p>Same per-frame content-mean replacement for every one of 48 spatial cells. Every prefix recomputed. Clip-wide coordinate scores, not pixel probabilities or temporal saliency. Positive=removal decreases spill margin; one shared signed scale.</p>'
    for row in summary:body+=f'<p>{html.escape(row["id"])}</p><img width="1280" src="{row["id"]}.png">'
    (output/'index.html').write_text(body+'</body></html>')
    (output/'summary.json').write_text(json.dumps(dict(scale=scale,rows=summary,scope='input_intervention',
        interpolation='nearest',opacity='0.4 * min(abs(delta_margin)/shared_scale,1)',zero_evidence='exactly transparent',
        native_scores_unchanged=True,per_frame_normalization=False),indent=2)+'\n')
    print(json.dumps(dict(scale=scale,sources=len(summary))),flush=True)


if __name__=='__main__':main()
