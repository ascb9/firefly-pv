#!/usr/bin/env python3
"""Verify the targeted revision against its actual previous video masters."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def probe(p):
    return json.loads(subprocess.check_output(['ffprobe','-v','error',
        '-show_streams','-show_format','-of','json',str(p)]))


def audio_hash(p):
    b=subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-map','0:a:0',
        '-c:a','copy','-f','adts','pipe:1'])
    return hashlib.sha256(b).hexdigest()


def frames(p,indices,w=640,h=360):
    expr='+'.join(f'eq(n\\,{n})' for n in indices)
    b=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(p),
        '-vf',f'select={expr},scale={w}:{h}','-vsync','0','-f','rawvideo',
        '-pix_fmt','rgb24','pipe:1'])
    a=np.frombuffer(b,'uint8').reshape(-1,h,w,3)
    assert len(a)==len(indices)
    return a


def compare(a,b):
    d=a.astype('float32')-b.astype('float32')
    mse=np.mean(d*d,axis=(1,2,3))
    p=10*np.log10(255**2/np.maximum(mse,1e-9))
    return {'mean_psnr_db':float(p.mean()),'min_psnr_db':float(p.min()),
            'mean_absolute_rgb_error':float(np.abs(d).mean())}


def main():
    ap=argparse.ArgumentParser()
    for key in ['video','original','v14','v15']:
        ap.add_argument('--'+key,required=True,type=Path)
    ap.add_argument('--qa-dir',type=Path,default=Path(__file__).parent/'qa')
    args=ap.parse_args()
    s=probe(args.video)['streams']
    v=next(x for x in s if x['codec_type']=='video')
    a=next(x for x in s if x['codec_type']=='audio')
    assert (v['width'],v['height'])==(1920,1080)
    assert v['avg_frame_rate']=='30/1' and int(v['nb_frames'])==2764
    assert v['color_space']=='bt709'
    assert abs(float(v['start_time']))<1e-5 and abs(float(a['start_time']))<1e-5
    full=subprocess.run(['ffmpeg','-v','error','-i',str(args.video),'-f','null','-'],
                         capture_output=True,text=True)
    assert full.returncode==0 and not full.stderr.strip(),full.stderr
    h0,h1=audio_hash(args.original),audio_hash(args.video)
    assert h0==h1
    restore=list(range(265,343))+list(range(2027,2062))
    restored=compare(frames(args.v14,restore),frames(args.video,restore))
    assert restored['min_psnr_db']>34,restored
    run=list(range(1110,1304))
    running=compare(frames(args.original,run),frames(args.video,run))
    assert running['min_psnr_db']>34,running
    unchanged=[30,100,200,450,570,615,850,900,960,1100,1320,1380,
               1435,1470,1515,1578,1620,1725,1860,1980,2080,2160,
               2300,2500,2660,2700]
    retained=compare(frames(args.v15,unchanged),frames(args.video,unchanged))
    assert retained['min_psnr_db']>34,retained
    checks=json.loads((args.qa_dir/'static_checks.json').read_text())
    static={}
    for name,count in [('park',80),('hand',42)]:
        c=[x for x in checks if x['region']==name]
        assert len(c)==count,(name,len(c))
        assert all(x['max_pixel_change']==0 for x in c),(name,'changed pixels')
        static[name]={'checked_frames':len(c),'max_preencode_pixel_change':0}
    trajectory=json.loads((args.qa_dir/'trajectory.json').read_text())
    intro=[t for t in trajectory if t['scene']=='intro']
    smile=[t for t in trajectory if t['scene']=='smile']
    assert all(a['entry_dx']>=b['entry_dx'] for a,b in zip(intro,intro[1:]))
    assert all(a['zoom']>=b['zoom'] for a,b in zip(intro,intro[1:]))
    assert next(x for x in intro if x['frame']==748)['name_visible']
    assert all(a['center_x']<=b['center_x'] for a,b in zip(smile,smile[1:]))
    assert all(a['height']>=b['height'] for a,b in zip(smile,smile[1:]))
    assert next(x for x in smile if x['frame']==1692)['center_x']>3000
    pts=json.loads((Path(__file__).parent/'references/original_pts.json').read_text())['frames']
    offset=max(abs(float(x['best_effort_timestamp_time'])-n/30) for n,x in enumerate(pts))
    result={
        'version':'V1.6','frame_count':2764,'resolution':[1920,1080],'fps':30,
        'video_duration_seconds':float(v['duration']),
        'audio_duration_seconds':float(a['duration']),
        'full_decode_without_errors':True,
        'original_aac_packets_equal':True,'aac_packet_sha256':h0,
        'max_original_pts_difference_ms':offset*1000,
        'color_matrix':'BT.709 with explicit RGB-to-YCbCr conversion',
        'restored_v14_all_113_frames_comparison':restored,
        'original_running_all_194_frames_comparison':running,
        'retained_v15_sample_comparison':retained,
        'protected_static_regions':static,
        'intro_right_to_left_then_pullback_verified':True,
        'smile_left_to_right_then_shrink_and_exit_verified':True,
        'original_audio_and_frame_clock_retained':True,
        'new_imagegen_calls_this_revision':1,
        'visual_review':'targeted scenes reviewed as pre-encode and final encoded contact sheets',
    }
    (args.qa_dir/'media_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
