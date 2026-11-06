#!/usr/bin/env python3
"""Audit every rendered cell and export real sampled frames without refitting.

This is rank-zero artifact consolidation of the completed DDP evaluation. The
saved class-target key gradients remain conditional explanations, not pixel
probabilities. No contextual class-vote map is painted over a source image.
"""
from __future__ import annotations
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
import torch.nn.functional as F
from vision_model.data.sampling import TemporalSampler,TemporalSamplingConfig
from vision_model.interpretability.map_geometry import (spatial_evidence_overlay,
    inverse_letterbox_map,letterbox_bounds,nonnegative_evidence_colors)
from train_spatial_semantic import DEFAULT_TEACHER
from diagnose_spill_representation import digest
from visualize_trajectory_video import decode_selected_frames


def evidence_peak(grid,support,height,width):
    """Find the numerical maximum over content cells; never use a spill ROI."""
    values=grid.flatten().masked_fill(~support.flatten(),-torch.inf)
    index=int(values.argmax());value=float(values[index])
    if not np.isfinite(value) or value<=0:return None
    row,column=divmod(index,grid.shape[1])
    top,left,h,w=letterbox_bounds(height,width)
    x=((column+.5)*256/grid.shape[1]-left)/w*width
    y=((row+.5)*256/grid.shape[0]-top)/h*height
    return dict(native_row_col=[row,column],value=value,
        source_center_xy=[min(max(x,0),width-1),min(max(y,0),height-1)],
        tied_maximum_cells=int((values==value).sum()))


def render_blue_red_frame(original,grids,*,support,scale,font,title,detail,predictions):
    """Display original, two overlays and two raw fields on one fixed scale.

    The white cross marks the numerical maximum, including failure cases. Low
    values are blue; this is not a verified background/foreground assignment.
    """
    source_width,source_height=original.size
    width,height=320,240
    display=original.resize((width,height),Image.Resampling.BILINEAR)
    pixels=torch.from_numpy(np.asarray(display).copy())
    canvas=Image.new('RGB',(5*width,392),'white');draw=ImageDraw.Draw(canvas)
    draw.text((6,3),title,fill='black',font=font)
    draw.text((6,23),'Offline episode target: conditional key sensitivity, not input-pixel attribution or a spill probability.',fill='black',font=font)
    draw.text((6,43),detail,fill='black',font=font)
    headings=('Original','Frozen gradient overlay','Trained gradient overlay','Frozen raw gradient','Trained raw gradient')
    for column,heading in enumerate(headings):draw.text((column*width+6,63),heading,fill='black',font=font)
    canvas.paste(display,(0,80));peaks={}
    for j,(name,grid) in enumerate(grids.items()):
        overlay=spatial_evidence_overlay(pixels,grid,scale=scale,alpha=.4,
            scope='conditional_attention_key_gradient',signed=False,style='blue_red')
        field=inverse_letterbox_map(grid,height,width,interpolation='nearest')
        heatmap=nonnegative_evidence_colors(field,scale=scale)
        canvas.paste(Image.fromarray(overlay.numpy()),((j+1)*width,80))
        canvas.paste(Image.fromarray(heatmap.numpy()),((j+3)*width,80))
        peak=evidence_peak(grid,support,source_height,source_width)
        peaks[name]=peak
        if peak:
            x,y=peak['source_center_xy'];x=x/source_width*width;y=y/source_height*height
            for column in (j+1,j+3):
                cx,cy=column*width+x,80+y
                draw.line((cx-6,cy,cx+6,cy),fill='black',width=5)
                draw.line((cx,cy-6,cx,cy+6),fill='black',width=5)
                draw.line((cx-5,cy,cx+5,cy),fill='white',width=1)
                draw.line((cx,cy-5,cx,cy+5),fill='white',width=1)
        text=f'{name}: peak={peak["value"]:.6g} ({peak["value"]/scale:.3f} of shared scale)' if peak else f'{name}: no positive value'
        draw.text((6+j*800,323),text,fill='black',font=font)
    legend=nonnegative_evidence_colors(torch.linspace(0,scale,320)[None].expand(12,-1),scale=scale)
    canvas.paste(Image.fromarray(legend.numpy()),(6,347))
    draw.text((335,345),f'0 (blue) to {scale:.8g} (red); cross = maximum, not verified spill. Alpha 0.4.',fill='black',font=font)
    draw.text((6,365),'Fixed scale across all sources, models and times; native cells; no smoothing, ROI or per-frame normalization.',fill='black',font=font)
    draw.text((6,380),f'Class predictions: {predictions}. A correct class does not validate the highlighted position.',fill='black')
    return canvas,peaks


def encode_video(destination,frames,fps):
    size=f'{frames[0].width}x{frames[0].height}'
    command=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',size,
             '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast',
             '-crf','18','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',
             '-n',str(destination)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for frame in frames:process.stdin.write(frame.tobytes())
        process.stdin.close()
        error=process.stderr.read().decode();status=process.wait()
        if status:raise RuntimeError(error)
    except BaseException:
        if process.poll() is None:process.kill();process.wait()
        raise
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
        '-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_read_frames,avg_frame_rate',
        '-of','json',str(destination)],text=True))['streams'][0]
    if info['codec_name']!='h264' or int(info['nb_read_frames'])!=len(frames):
        raise RuntimeError(f'Incorrect encoded frames: {destination}')
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figures-name',default='figures_v2')
    parser.add_argument('--teacher-cache',type=Path,default=DEFAULT_TEACHER)
    parser.add_argument('--style',choices=('magnitude','blue_red'),default='magnitude',
        help='Explicit display convention; blue_red colors low gradients blue without changing raw scores')
    args=parser.parse_args();torch.set_num_threads(4)
    if int(os.environ.get('RANK','0'))!=0:raise RuntimeError('Artifact consolidation must run on rank zero')
    figures=args.run/args.figures_name
    manifest=json.loads((figures/'manifest.json').read_text())
    if manifest['contextual_feature_contributions_are_localization_overlays']:
        raise ValueError('Contextual class votes must not be image-localization overlays')
    if manifest['zero_evidence']!='exactly transparent' or manifest['interpolation']!='nearest native cells':
        raise ValueError('Incorrect overlay contract')
    raw=torch.load(args.teacher_cache,map_location='cpu',weights_only=False)
    attention_path=args.run/'spatial_evaluation/attention_maps.pt'
    if digest(attention_path)!=manifest['attention_sha256']:raise ValueError('Attribution file changed')
    attention=torch.load(attention_path,map_location='cpu',weights_only=False)
    indexed={(row['case'],row['index']):row for row in attention['rows']}
    names=('FT0_frozen','FT1_attention');representations={}
    for name in names:
        if digest(args.run/name/'checkpoint.pt')!=manifest['checkpoint_sha256'][name]:
            raise ValueError('Rendered checkpoint does not match current checkpoint')
        saved=torch.load(args.run/name/'representations.pt',map_location='cpu',weights_only=False,mmap=True)
        if saved['ids']!=raw['ids']:raise ValueError('Source identity mismatch')
        representations[name]=saved
    if attention['ids']!=raw['ids'] or [row['id'] for row in manifest['rows']]!=raw['ids']:
        raise ValueError('Incomplete or duplicated attribution/figure sources')
    expected={(name,i) for name in names for i in range(len(raw['ids']))}
    if set(indexed)!=expected or len(attention['rows'])!=len(expected):
        raise ValueError('Repeated or missing evaluated source')
    scale=manifest['scales']['positive_gradient'];alpha=manifest['maximum_alpha']
    actual_max=max(float(row['maps']['positive_gradient'].max()) for row in attention['rows'])
    if scale!=actual_max or alpha!=.4:raise ValueError('Shared scale/opacity mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    for directory in ('videos','storyboards','thumbnails'):(args.output/directory).mkdir()
    records={r['video_id']:r for r in (json.loads(line) for line in Path(raw['manifest']).read_text().splitlines())}
    try:font=ImageFont.truetype('DejaVuSans.ttf',13)
    except OSError:font=ImageFont.load_default()
    rows=[];maximum_pixel_error=0;maximum_embedding_error=0.;maximum_margin_error=0.
    for i,source_id in enumerate(raw['ids']):
        record=records[source_id];source=Path(raw['dataset_root'])/record['video_path']
        if digest(source)!=raw['video_sha256'][i]:raise ValueError('Video changed')
        sampled,frame_valid=TemporalSampler(TemporalSamplingConfig(**raw['sampling'])).sample_with_validity(
            num_frames=record['num_frames'],source_fps=record['fps'])
        n=int(raw['valid'][i].sum());indices=sampled[:2*n].tolist()
        if not frame_valid[:2*n].all() or indices[1::2]!=raw['frame_indices'][i,:n].tolist():
            raise ValueError('Invalid frame/tubelet indexing')
        if manifest['rows'][i]['frame_indices']!=indices[1::2]:raise ValueError('Figure time index mismatch')
        decoded=decode_selected_frames(source,indices,frame_count=record['num_frames'],width=record['width'],height=record['height'])
        board=Image.open(figures/'sources'/f'{source_id}.png').convert('RGB')
        if board.size!=(1200,55+n*205):raise ValueError('Unexpected audited board geometry')
        pixel_error=0
        for t,index in enumerate(indices[1::2]):
            display=Image.fromarray(decoded[index]).resize((240,180))
            source_pixels=torch.from_numpy(np.asarray(display).copy())
            actual_original=np.asarray(board.crop((0,55+t*205,240,55+t*205+180)))
            if not np.array_equal(actual_original,source_pixels.numpy()):raise ValueError('Displayed original differs from source')
            for column,name in enumerate(names,1):
                grid=indexed[(name,i)]['maps']['positive_gradient'][t].reshape(16,16)
                predicted=spatial_evidence_overlay(source_pixels,grid,scale=scale,alpha=alpha,
                    scope='conditional_attention_key_gradient',signed=False).numpy()
                observed=np.asarray(board.crop((column*240,55+t*205,(column+1)*240,55+t*205+180)))
                error=int(np.abs(predicted.astype(int)-observed.astype(int)).max())
                pixel_error=max(pixel_error,error)
        if pixel_error:raise RuntimeError(f'Overlay does not implement declared values: {source_id}, {pixel_error}')
        maximum_pixel_error=max(maximum_pixel_error,pixel_error)
        replay={};predictions={}
        for name,saved in representations.items():
            support=saved['content'][i]
            q=saved['patch_features'][i,0,:n][:,support]
            z=F.normalize(q.mean((0,1)),dim=0)
            embedding_error=float((z-saved['embeddings'][i,0]).abs().max())
            scores=z@saved['prototypes'].T;margin=float(scores[1]-scores[0])
            margin_error=abs(float(saved['signed_maps'][i,0,:n][:,support].mean())-margin)
            if max(embedding_error,margin_error)>2e-5:raise ValueError('Saved classifier/margin does not reconstruct')
            replay[name]=dict(embedding_max_error=embedding_error,margin_reconstruction_error=margin_error,
                live_checkpoint_replay_max_error=indexed[(name,i)]['checkpoint_replay_max_error'])
            maximum_embedding_error=max(maximum_embedding_error,embedding_error)
            maximum_margin_error=max(maximum_margin_error,margin_error)
            predictions[name]=int(scores.argmax())
        split=manifest['rows'][i]['split'];frames=[];frame_peaks=[]
        for position,index in enumerate(indices):
            t=position//2
            if args.style=='blue_red':
                grids={name:indexed[(name,i)]['maps']['positive_gradient'][t].reshape(16,16) for name in names}
                canvas,peaks=render_blue_red_frame(Image.fromarray(decoded[index]),grids,
                    support=representations[names[0]]['content'][i],scale=scale,font=font,
                    title=f'{source_id} | {split} | label={"cold spill" if int(raw["labels"][i]) else "clean"}',
                    detail=f'Frame {index} at {index/record["fps"]:.3f}s | key tubelet {t+1} | shared scale {scale:.8g}',
                    predictions=predictions)
                frames.append(canvas);frame_peaks.append(peaks)
                continue
            original=Image.fromarray(decoded[index]).resize((384,288),Image.Resampling.BILINEAR)
            source_pixels=torch.from_numpy(np.asarray(original).copy())
            canvas=Image.new('RGB',(1152,400),'white');draw=ImageDraw.Draw(canvas)
            draw.text((6,3),f'{source_id} | {split} | label={"cold spill" if int(raw["labels"][i]) else "clean"}',fill='black',font=font)
            draw.text((6,23),'Offline episode target; conditional attention-key gradients, not an input-pixel spill mask',fill='black',font=font)
            draw.text((6,43),f'Frame {index} at {index/record["fps"]:.3f}s | key tubelet {t+1} | shared scale {scale:.8g}',fill='black',font=font)
            for column,title in enumerate(('Original','Frozen attention','Trained final attention')):
                draw.text((column*384+6,63),title,fill='black',font=font)
            canvas.paste(original,(0,80))
            for column,name in enumerate(names,1):
                grid=indexed[(name,i)]['maps']['positive_gradient'][t].reshape(16,16)
                overlay=spatial_evidence_overlay(source_pixels,grid,scale=scale,alpha=alpha,
                    scope='conditional_attention_key_gradient',signed=False)
                canvas.paste(Image.fromarray(overlay.numpy()),(column*384,80))
            draw.text((6,372),'Zero = original RGB; red opacity = 0.4*gradient/shared scale. No blur or per-frame normalization.',fill='black',font=font)
            draw.text((6,386),f'Class predictions: frozen={predictions[names[0]]}, trained={predictions[names[1]]}; these do not validate spill location.',fill='black')
            frames.append(canvas)
        video_info=encode_video(args.output/'videos'/f'{source_id}.mp4',frames,raw['sampling']['target_fps'])
        frames[-1].save(args.output/'thumbnails'/f'{source_id}.png')
        if args.style=='blue_red':
            storyboard=Image.new('RGB',(frames[0].width,frames[0].height*n),'white')
            for t in range(n):storyboard.paste(frames[2*t+1],(0,t*frames[0].height))
            storyboard.save(args.output/'storyboards'/f'{source_id}.png')
        else:shutil.copyfile(figures/'sources'/f'{source_id}.png',args.output/'storyboards'/f'{source_id}.png')
        rows.append(dict(id=source_id,split=split,label=int(raw['labels'][i]),source_sha256=raw['video_sha256'][i],
            sampled_frame_indices=indices,timestamps=[j/record['fps'] for j in indices],predictions=predictions,
            original_v2_overlay_max_pixel_error=pixel_error,replay=replay,video=video_info,
            numerical_peaks_per_sampled_frame=frame_peaks))
        print(json.dumps(dict(completed=len(rows),total=len(raw['ids']),id=source_id)),flush=True)
    result=dict(rows=rows,sources=len(rows),frame_count=sum(int(r['video']['nb_read_frames']) for r in rows),
        maximum_overlay_pixel_error=maximum_pixel_error,maximum_embedding_replay_error=maximum_embedding_error,
        maximum_margin_reconstruction_error=maximum_margin_error,source_rank='rank-zero consolidation; DDP evaluation assigned each source once',
        scope='conditional_attention_key_gradient',style=args.style,
        zero='blue (low sensitivity), not verified clean background' if args.style=='blue_red' else 'exactly transparent',
        interpolation='nearest native cells',palette='blue/cyan/green/yellow/red' if args.style=='blue_red' else 'red',
        opacity='fixed 0.4' if args.style=='blue_red' else '0.4 * min(positive_gradient / shared_global_scale, 1)',
        peak_marker='automatic numerical maximum over content cells' if args.style=='blue_red' else None,
        shared_scale=scale,checkpoints=manifest['checkpoint_sha256'],
        attention_sha256=manifest['attention_sha256'],teacher_cache_sha256=digest(args.teacher_cache),
        exporter_sha256=digest(Path(__file__)),geometry_sha256=digest(ROOT/'src/vision_model/interpretability/map_geometry.py'),
        classifier_and_raw_maps_changed=False,localization_validated=False)
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    convention='Low sensitivity is blue, high sensitivity is red; fixed alpha 0.4. Raw fields appear beside the overlays. Crosses mark numerical maxima, not verified spill positions.' if args.style=='blue_red' else 'Zero relevance is transparent; red opacity follows sensitivity magnitude.'
    body=f'<html><head><meta charset="utf-8"><title>80 videos: corrected overlay</title></head><body><h1>80-video comparison: {args.style}</h1><p>Actual sampled frames, frozen versus trained final attention. {convention} One scale for all models, sources and times. These are conditional key gradients, not validated spill masks. Episode-wide contextual class votes are excluded.</p>'
    for row in rows:
        name=html.escape(row['id'])
        body+=f'<p>{name} | {row["split"]}</p><video controls preload="none" width="960" poster="thumbnails/{name}.png"><source src="videos/{name}.mp4" type="video/mp4"></video><p><a href="storyboards/{name}.png">Native-grid storyboard</a></p>'
    (args.output/'index.html').write_text(body+'</body></html>')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
