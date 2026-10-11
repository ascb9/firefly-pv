#!/usr/bin/env python3
"""Inspect the decoded delivered MP4, rather than only renderer stills."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
if (HERE/'deps').is_dir():sys.path.insert(0,str(HERE/'deps'))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from shot54 import PATCHES,COUNT,FPS,dump
cv2.setNumThreads(2)

def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams',
        '-show_format','-of','json',str(path)]))

def get_frames(path,indices):
    cap=cv2.VideoCapture(str(path));out={}
    last=-1
    for n in sorted(set(indices)):
        if last<0 or n-last>90:
            cap.set(cv2.CAP_PROP_POS_FRAMES,n);last=n-1
        while last<n:
            ok,im=cap.read();last+=1
            if not ok:raise RuntimeError(f'Cannot decode {path} frame {n}')
        out[n]=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    cap.release();return out

def audio_hash(path):
    data=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:a:0',
         '-c','copy','-f','adts','pipe:1'])
    return hashlib.sha256(data).hexdigest()

def contact_sheet(candidate, original, frames, start, out, label):
    width=1280;tw,th=320,180
    cols=2;rows=math.ceil(len(frames)/cols)
    sheet=Image.new('RGB',(width,rows*225),(16,22,30));d=ImageDraw.Draw(sheet)
    font=ImageFont.truetype(str(HERE/'assets/NotoSansCJKsc-Regular.otf'),17)
    ca=get_frames(candidate,[n-start for n in frames]);og=get_frames(original,frames)
    for j,n in enumerate(frames):
        x=(j%cols)*640;y=(j//cols)*225
        d.text((x+8,y+4),f'{n/FPS:.3f}s  /  frame {n}   原PV | {label}',font=font,fill='white')
        for k,im in enumerate((og[n],ca[n-start])):
            sheet.paste(Image.fromarray(im).resize((tw,th),Image.Resampling.LANCZOS),(x+k*tw,y+29))
    sheet.save(out,quality=95)

def comparison_video(original,candidate,out):
    # Three actual MP4 segments, then the same portrait segment at half speed.
    cap0=cv2.VideoCapture(str(original));cap1=cv2.VideoCapture(str(candidate))
    font=ImageFont.truetype(str(HERE/'assets/NotoSansCJKsc-Regular.otf'),24)
    small=ImageFont.truetype(str(HERE/'assets/NotoSansCJKsc-Regular.otf'),21)
    encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1920x600','-r','30','-i','pipe:0','-an','-vf',
        'scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p',
        '-c:v','libx264','-crf','17','-preset','fast','-threads','4',
        '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',
        '-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    count=0
    try:
        for a,b,repeat,title in [(1296,1348,1,'腿脚静止'),(1410,1453,1,'柔和近景转场'),
                                (1635,1708,1,'表情切换与离场'),(1635,1708,2,'表情切换与离场 · 0.5倍速')]:
            cap0.set(cv2.CAP_PROP_POS_FRAMES,a);cap1.set(cv2.CAP_PROP_POS_FRAMES,a)
            for n in range(a,b):
                frames=[]
                for cap in (cap0,cap1):
                    ok,im=cap.read()
                    if not ok:raise RuntimeError(f'Comparison source ended at {n}')
                    frames.append(cv2.cvtColor(cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
                canvas=Image.new('RGB',(1920,600),(17,23,33));draw=ImageDraw.Draw(canvas)
                draw.text((24,16),'原始PV',font=font,fill=(245,245,245))
                draw.text((984,16),'V1.7 · 流萤角色替换',font=font,fill=(145,235,226))
                draw.text((460,18),f'{n/30:.3f}秒 · {title}',font=small,fill=(210,218,230))
                for k,im in enumerate(frames):canvas.paste(Image.fromarray(im),(k*960,60))
                raw=canvas.tobytes()
                for _ in range(repeat):encoder.stdin.write(raw);count+=1
        encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('Comparison encoding failed')
    finally:
        for cap in (cap0,cap1):cap.release()
        if encoder.poll() is None:encoder.terminate()
    return dict(frames=count,seconds=count/30,resolution=[1920,600],audio='muted review',
                left='original MP4',right='decoded V1.7 MP4',half_speed_portrait=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True,type=Path)
    ap.add_argument('--original',required=True,type=Path);ap.add_argument('--base',type=Path)
    ap.add_argument('--candidate-start',type=int,default=0)
    ap.add_argument('--output-dir',required=True,type=Path);ap.add_argument('--full',action='store_true')
    ap.add_argument('--comparison',type=Path)
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    ca_probe=probe(args.candidate);video=next(s for s in ca_probe['streams'] if s['codec_type']=='video')
    full=args.full
    frames=[1643,1646,1649,1652,1655,1660,1661,1664,1667,1670,1675,1680,1683,1686,1688,1690,1691,1696,1699,1709]
    contact_sheet(args.candidate,args.original,frames,args.candidate_start,
                  args.output_dir/'encoded_portrait_comparison.jpg','V1.7')
    result=dict(version='V1.7',candidate_filename=args.candidate.name,
        video=dict(width=video['width'],height=video['height'],
            fps=video['r_frame_rate'],frame_count=int(video['nb_frames']),
            duration=float(video['duration']),color_space=video.get('color_space')),
        inspection='decoded MP4 side-by-side frame sheets',visual_review_completed=False)
    if full:
        assert int(video['nb_frames'])==COUNT and video['width']==1920 and video['height']==1080
        assert video['r_frame_rate']=='30/1' and video['color_space']=='bt709'
        subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(args.candidate),
            '-map','0:v:0','-map','0:a:0','-f','null','-'],check=True)
        result['full_decode_without_errors']=True
        ah=audio_hash(args.candidate);oh=audio_hash(args.original)
        assert ah==oh
        result['original_aac_packets_equal']=True;result['aac_packet_sha256']=ah
        contact_sheet(args.candidate,args.original,[1303,1304,1314,1330,1341,1342],0,
            args.output_dir/'encoded_feet_comparison.jpg','V1.7')
        contact_sheet(args.candidate,args.original,[1417,1418,1421,1424,1427,1430,1437,1441,1442,1445],0,
            args.output_dir/'encoded_kiss_comparison.jpg','V1.7')
        trajectory=json.loads((args.output_dir/'trajectory.json').read_text())
        result['portrait_similarity_transform']=dict(frames=len(trajectory),
            max_axis_scale_difference=max(r['axis_scale_difference'] for r in trajectory),
            max_axes_dot_product=max(abs(r['axes_dot_product']) for r in trajectory),
            open_to_closed_cut_frame=1661,original_empty_floor_start_frame=1691)
        checks=json.loads((args.output_dir/'static_feet.json').read_text())
        assert len(checks)==38 and max(x['actual_pre_encode_max_pixel_delta'] for x in checks)==0
        result['static_feet_preencode']=dict(frames=38,max_pixel_change=0,region=[0,0,1920,900])
        fs=get_frames(args.candidate,list(range(1304,1342)))
        master=cv2.cvtColor(cv2.resize(fs[1304][:900],(960,450)),cv2.COLOR_RGB2GRAY).astype(np.float32)
        window=cv2.createHanningWindow((960,450),cv2.CV_32F);drifts=[];errors=[]
        for n,im in fs.items():
            g=cv2.cvtColor(cv2.resize(im[:900],(960,450)),cv2.COLOR_RGB2GRAY).astype(np.float32)
            shift,response=cv2.phaseCorrelate(master.copy(),g.copy(),window)
            drifts.append(math.hypot(*shift)*2)
            errors.append(float(np.abs(im[:900].astype(np.float32)-fs[1304][:900]).mean()))
        result['static_feet_encoded']=dict(frames=len(fs),
            max_estimated_translation_pixels=max(drifts),mean_rgb_difference=float(np.mean(errors)),
            method='phase correlation and RGB differences on decoded art area; H.264 changes allowed')
        if args.base:
            ns=[0,264,265,300,342,343,508,653,675,719,724,765,812,813,983,1020,1062,
                1110,1170,1290,1303,1342,1390,1417,1442,1490,1605,1637,1710,
                1950,2027,2038,2048,2061,2218,2240,2271,2592,2763]
            ca=get_frames(args.candidate,ns);bs=get_frames(args.base,ns);psnrs=[];mad=[]
            for n in ns:
                assert not any(a<=n<b for _,a,b in PATCHES)
                delta=ca[n].astype(np.float32)-bs[n].astype(np.float32)
                mse=float(np.mean(delta*delta));psnrs.append(10*math.log10(255**2/max(mse,1e-8)))
                mad.append(float(np.abs(delta).mean()))
            result['retained_v16_decoded_sample_comparison']=dict(frames=ns,
                mean_psnr_db=float(np.mean(psnrs)),min_psnr_db=min(psnrs),
                mean_absolute_rgb_error=float(np.mean(mad)),
                note='all 2630 outside-patch frames enter the encoder unchanged; differences are generation-loss encoding')
        if args.comparison:
            args.comparison.parent.mkdir(parents=True,exist_ok=True)
            result['comparison_video']=comparison_video(args.original,args.candidate,args.comparison)
    dump(args.output_dir/'validation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
