#!/usr/bin/env python3
"""Render every source with shared scales; summarize bounded causal experiment."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE,trustworthiness
from sklearn.metrics import silhouette_score
from PIL import Image,ImageDraw,ImageFont
from vision_model.interpretability.map_geometry import inverse_letterbox_map,spatial_evidence_overlay
from train_spatial_semantic import DEFAULT_TEACHER
from visualize_trajectory_video import decode_selected_frames
from diagnose_spill_representation import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output-name',default='figures_v2')
    a=p.parse_args();torch.set_num_threads(4)
    out=a.run/a.output_name;out.mkdir(exist_ok=False);(out/'sources').mkdir();(out/'audit_fields').mkdir()
    raw=torch.load(DEFAULT_TEACHER,map_location='cpu',weights_only=False)
    attention=torch.load(a.run/'spatial_evaluation/attention_maps.pt',map_location='cpu',weights_only=False)
    names=['FT0_frozen','FT1_attention'];cases={}
    scales=dict(raw=0.,positive_gradient=0.,signed=0.)
    indexed={(r['case'],r['index']):r for r in attention['rows']}
    for name in names:
        rep=torch.load(a.run/name/'representations.pt',map_location='cpu',weights_only=False)
        cases[name]=rep
        scales['signed']=max(scales['signed'],float(rep['signed_maps'].abs().max()))
    for r in attention['rows']:
        for kind in ('raw','positive_gradient'):scales[kind]=max(scales[kind],float(r['maps'][kind].max()))
    def color(grid,kind,h,w):
        x=inverse_letterbox_map(grid.reshape(16,16),h,w,interpolation='nearest').numpy()/max(scales[kind],1e-30)
        if kind=='signed':colors=plt.get_cmap('bwr')((np.clip(x,-1,1)+1)/2)[...,:3]
        else:colors=plt.get_cmap('turbo')(np.clip(x,0,1))[...,:3]
        return torch.from_numpy(np.round(colors*255).astype(np.uint8))
    records={r['video_id']:r for r in (json.loads(l) for l in Path(raw['manifest']).read_text().splitlines())}
    try:font=ImageFont.truetype('DejaVuSans.ttf',12)
    except OSError:font=ImageFont.load_default()
    gallery=[];cold_contacts=[];comparisons=[]
    for i,source_id in enumerate(raw['ids']):
        record=records[source_id];source=Path(raw['dataset_root'])/record['video_path']
        if digest(source)!=raw['video_sha256'][i]:raise ValueError('Source changed')
        ends=raw['frame_indices'][i][raw['valid'][i]].tolist();n=len(ends)
        decoded=decode_selected_frames(source,ends,frame_count=record['num_frames'],width=record['width'],height=record['height'])
        split='TRAIN' if i in cases[names[0]]['train_indices'] else 'SOURCE VALIDATION'
        label='cold spill' if int(raw['labels'][i]) else 'clean'
        width,height=240,180
        board=Image.new('RGB',(5*width,55+n*(height+25)),'white');draw=ImageDraw.Draw(board)
        draw.text((6,3),f'{source_id} | {split} | {label} | CONDITIONAL KEY GRADIENTS; not input masks | native cells, opacity by magnitude',fill='black',font=font)
        headings=['Original','FT0 gradient overlay','FT1 gradient overlay','FT0 native gradient','FT1 native gradient']
        for col,title in enumerate(headings):draw.text((col*width+4,29),title,fill='black',font=font)
        audit=Image.new('RGB',(4*width,55+n*(height+25)),'white');adraw=ImageDraw.Draw(audit)
        adraw.text((5,3),f'{source_id}: RAW/CONTEXTUAL DIAGNOSTICS; no localization overlays',fill='black',font=font)
        for col,title in enumerate(['FT0 raw key attention','FT1 raw key attention','FT0 episode feature votes','FT1 episode feature votes']):adraw.text((col*width+4,29),title,fill='black',font=font)
        thumbnail=None
        for t,index in enumerate(ends):
            frame=torch.from_numpy(decoded[index].copy());h,w=frame.shape[:2]
            originals=Image.fromarray(frame.numpy()).resize((width,height))
            display=torch.from_numpy(np.asarray(originals).copy())
            images=[originals];native=[]
            for name in names:
                grid=indexed[(name,i)]['maps']['positive_gradient'][t].reshape(16,16)
                overlay=spatial_evidence_overlay(display,grid,scale=scales['positive_gradient'],
                    scope='conditional_attention_key_gradient',signed=False,alpha=.4)
                images.append(Image.fromarray(overlay.numpy()))
                native.append(Image.fromarray(color(grid,'positive_gradient',height,width).numpy()))
            images.extend(native)
            for col,(kind,name) in enumerate((('raw',names[0]),('raw',names[1]),('signed',names[0]),('signed',names[1]))):
                grid=cases[name]['signed_maps'][i,0,t] if kind=='signed' else indexed[(name,i)]['maps'][kind][t]
                field=Image.fromarray(color(grid,kind,height,width).numpy())
                audit.paste(field,(col*width,55+t*(height+25)))
            for col,image in enumerate(images):board.paste(image,(col*width,55+t*(height+25)))
            draw.text((5,55+t*(height+25)+height+3),f'Source frame {index}, {index/record["fps"]:.3f}s | key tubelet {t+1}; past keys remain at their own times',fill='black',font=font)
            if t==n-1:
                thumbnail=Image.new('RGB',(4*width,height+25),'white');td=ImageDraw.Draw(thumbnail)
                for col,j in enumerate((0,1,2,4)):thumbnail.paste(images[j],(col*width,25))
                td.text((5,3),f'{source_id} | {split} | original / FT0 conditional gradient / FT1 conditional gradient / FT1 native grid',fill='black',font=font)
        board.save(out/'sources'/f'{source_id}.png')
        audit.save(out/'audit_fields'/f'{source_id}.png')
        thumbnail.save(out/'sources'/f'{source_id}_last.png')
        gallery.append(dict(id=source_id,split=split,label=label,image=f'sources/{source_id}.png',thumbnail=f'sources/{source_id}_last.png',
            audit_fields=f'audit_fields/{source_id}.png',frame_indices=ends,timestamps=[j/record['fps'] for j in ends],source_sha256=raw['video_sha256'][i]))
        if int(raw['labels'][i]) and split=='SOURCE VALIDATION':cold_contacts.append(thumbnail)
        support=cases[names[0]]['content'][i].repeat(n,1)
        row=dict(id=source_id,label=int(raw['labels'][i]),split=split)
        for kind in ('raw','positive_gradient','signed_attention_gradient'):
            x=indexed[(names[0],i)]['maps'][kind][support].double();y=indexed[(names[1],i)]['maps'][kind][support].double()
            row[kind+'_cosine']=float(torch.nn.functional.cosine_similarity(x,y,dim=0))
            row[kind+'_relative_L2']=float((x-y).norm()/x.norm().clamp_min(1e-30))
        row['positive_gradient_target_cosine_FT0']=indexed[(names[0],i)]['spill_clean_positive_gradient_cosine']
        row['positive_gradient_target_cosine_FT1']=indexed[(names[1],i)]['spill_clean_positive_gradient_cosine']
        comparisons.append(row)
    contact=Image.new('RGB',(cold_contacts[0].width,len(cold_contacts)*cold_contacts[0].height),'white')
    for j,panel in enumerate(cold_contacts):contact.paste(panel,(0,j*panel.height))
    contact.save(out/'all_8_validation_cold.png')
    body='<html><head><meta charset="utf-8"><title>Final attention: all 80 sources</title></head><body><h1>All 80 sources: corrected overlay logic</h1><p>Positive gradients are conditional attention-key sensitivity, not input-pixel attribution or full LeGrad. Zero relevance preserves original RGB; red opacity is 0.4*gradient/shared_scale. Native cells, no blur or per-frame normalization. Raw attention and contextual episode feature votes are separate audit fields, not localization overlays. For direct input interventions see ../../input_occlusion_figures_v2/index.html.</p>'
    for r in gallery:body+=f'<p>{html.escape(r["id"])} | {r["split"]} | {r["label"]} | <a href="{r["audit_fields"]}">raw/contextual audit fields</a></p><a href="{r["image"]}"><img width="960" src="{r["thumbnail"]}"></a>'
    (out/'index.html').write_text(body+'</body></html>')
    (out/'manifest.json').write_text(json.dumps(dict(rows=gallery,scales=scales,maximum_alpha=.4,zero_evidence='exactly transparent',
        opacity='0.4*min(positive_gradient/shared_global_scale,1)',interpolation='nearest native cells',
        contextual_feature_contributions_are_localization_overlays=False,raw_maps_unchanged=True,
        checkpoint_sha256={name:digest(a.run/name/'checkpoint.pt') for name in names},attention_sha256=digest(a.run/'spatial_evaluation/attention_maps.pt')),indent=2)+'\n')
    # Original-dimensional separation is reported alongside t-SNE distortion.
    fig,axes=plt.subplots(1,2,figsize=(12,5));tsne={}
    for ax,name in zip(axes,names):
        rep=cases[name];z=rep['embeddings'][:,0].numpy();labels=rep['labels'].numpy()
        xy=TSNE(n_components=2,perplexity=15,init='pca',learning_rate='auto',random_state=239,max_iter=1500).fit_transform(z)
        tsne[name]=dict(coordinates=xy.tolist(),original_cosine_silhouette=float(silhouette_score(z,labels,metric='cosine')),
            projected_silhouette=float(silhouette_score(xy,labels)),trustworthiness=float(trustworthiness(z,xy,n_neighbors=5,metric='cosine')),
            caveat='Transductive visualization of 64 training plus 16 selection sources, not an independent test; axis orientation arbitrary')
        for split,indices,marker in [('train',rep['train_indices'],'o'),('validation',rep['validation_indices'],'X')]:
            for label,col in ((0,'tab:blue'),(1,'tab:red')):
                ix=indices[rep['labels'][indices]==label]
                ax.scatter(xy[ix,0],xy[ix,1],marker=marker,c=col,label=f'{split} {"cold" if label else "clean"}',s=45)
        ax.set_title(name);ax.legend(fontsize=8);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('80 sources: t-SNE is a representation diagnostic, not spill localization');fig.tight_layout();fig.savefig(out/'tsne_80.png',dpi=150);plt.close(fig)
    (out/'tsne_80.json').write_text(json.dumps(tsne,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4));convergence={}
    for name in names:
        rows=list(csv.DictReader((a.run/name/'history.csv').open()));x=np.array([int(r['update']) for r in rows]);convergence[name]={}
        for ax,key in zip(axes,('loss','semantic_loss','local_feature_retention')):
            values=np.array([float(r[key]) for r in rows]);ax.plot(x,values,label=name);ax.set_title(key);ax.set_xlabel('Continuation update')
            # Prefix schedule has period six. Compare complete six-update
            # cycles to keep different sampled prefixes from biasing slopes.
            cycles=np.array([values[j:j+6].mean() for j in range(0,len(values)-5,6)])
            tail=cycles[-5:];slope=float(np.polyfit(np.arange(len(tail)),tail,1)[0]/6)
            convergence[name][key]=dict(final=float(values[-1]),final_6cycle_mean=float(cycles[-1]),last_30update_cycle_slope=slope,
                slope_fraction_of_tail_mean_per_update=slope/float(tail.mean()))
        axes[0].legend()
    fig.tight_layout();fig.savefig(out/'loss_curves.png',dpi=150);plt.close(fig)
    output=dict(map_comparisons=comparisons,convergence=convergence,tsne=tsne,scales=scales)
    controls_path=a.run/'spatial_evaluation/input_controls.json'
    if controls_path.exists():
        controls=json.loads(controls_path.read_text())['rows'];summary={}
        generator=np.random.default_rng(239)
        for name in names:
            summary[name]={}
            for subset,label in [('all',None),('cold',1),('clean',0)]:
                rows=[r for r in controls if r['case']==name and (label is None or r['label']==label)]
                diffs=[];entry={}
                for condition in rows[0]['conditions']:
                    entry[condition]=dict(accuracy=np.mean([r['conditions'][condition]['prediction']==r['label'] for r in rows]),
                        true_class_margin_drop=np.mean([(2*r['label']-1)*(r['conditions']['original']['margin']-r['conditions'][condition]['margin']) for r in rows]))
                for r in rows:
                    high=r['conditions']['original']['margin']-r['conditions']['high']['margin']
                    rand=np.mean([r['conditions']['original']['margin']-r['conditions'][f'random_{j}']['margin'] for j in range(3)])
                    diffs.append((2*r['label']-1)*(high-rand))
                diffs=np.array(diffs);means=diffs[generator.integers(0,len(diffs),size=(10000,len(diffs)))].mean(-1)
                entry['high_minus_random']=dict(mean=float(diffs.mean()),source_bootstrap_95ci=np.quantile(means,[.025,.975]).tolist(),n=len(rows))
                summary[name][subset]=entry
        output['input_control_summary']=summary
    (out/'analysis_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(figures=str(out),sources=len(gallery),scales=scales)),flush=True)


if __name__=='__main__':main()
