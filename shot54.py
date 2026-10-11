#!/usr/bin/env python3
"""V1.7: three frame-locked revisions on the validated V1.6 finished edit.

Reuse the three already approved portrait paintings. Camera movement uses
similarity transforms measured from the original PV, without facial warping.
"""
from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / 'deps').is_dir():
    sys.path.insert(0, str(HERE / 'deps'))
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
from captions import Captions
from compose import CelMotion, camera, ease, read_exact

cv2.setNumThreads(2)
W, H, FPS, COUNT = 1920, 1080, 30, 2764
PATCHES = [('static_feet', 1304, 1342), ('soft_kiss_push', 1418, 1442),
           ('portrait_rebuild', 1638, 1710)]
REVIEW = [1303,1304,1308,1314,1320,1330,1341,1342,1417,1418,1421,1424,
          1427,1430,1432,1437,1441,1442,1637,1638,1642,1643,1646,1649,
          1652,1655,1660,1661,1662,1664,1667,1670,1675,1680,1683,1686,
          1688,1690,1691,1695,1696,1699,1709,1710]
OPEN_ENTRY = [(-900,-402),(-894,-400),(-872,-388),(-806,-356),(-572,-248),
              (-368,-156),(-270,-112),(-206,-84),(-162,-64),(-130,-50),
              (-104,-40),(-84,-32),(-68,-26)]
CLOSED = [
 [1662,.9427102803973323,-.010427117775109542,21.977301122266653,59.30570926475203],
 [1663,.9526681133229786,-.010066282492624808,17.229334047361643,50.72755011470901],
 [1664,.9632962298345086,-.0075691467692787714,13.413787237531173,38.93640713701689],
 [1665,.9747386171177694,-.005220597353154249,9.214282435073788,26.732666110486885],
 [1666,.9869267995088927,-.002658262233855474,4.736111543726171,13.770839536603575],
 [1667,1,0,0,0],
 [1668,1.0140492520894064,.0028745909046971455,-5.173485597066269,-14.68244870955934],
 [1669,1.0290831227120918,.006028331064488091,-10.430333772372215,-30.346159859708784],
 [1670,1.0453155500415254,.0095046172321803,-16.119558154472664,-47.26492022616148],
 [1671,1.0627372983096262,.013143106978916862,-21.94840263323721,-65.28117149555374],
 [1672,1.0816373101808185,.01714023345299387,-28.243262061469107,-84.70039484974804],
 [1673,1.1024198741809847,.021469780198450764,-35.161401336569334,-105.75514461724168],
 [1674,1.125054798009436,.026211487588789672,-42.365562053310335,-128.61531242468536],
 [1675,1.150452543990827,.03134009310970507,-50.5555540218921,-153.7680050458709],
 [1676,1.1794462750359513,.03739573016770014,-59.7147077095491,-182.4323265383877],
 [1677,1.2134951146145836,.04445401731087804,-69.8603844913001,-215.80040232123548],
 [1678,1.2564703209048798,.052953416342155696,-82.99218487919197,-256.7676219979259],
 [1679,1.3041326473968462,.06316571614122125,-96.20919562273014,-302.3057715381619],
 [1680,1.3575849376654896,.0747409207541477,-110.04506338235194,-352.6698162748778],
 [1681,1.4195244024791271,.08807491664869367,-125.91422612544054,-409.80478766659],
 [1682,1.4902866121781555,.10286808565349212,-143.15767188375088,-473.3600781600624],
 [1683,1.5722568994203838,.12007236212305845,-161.67450331999362,-545.1775863732809],
 [1684,1.6696748923049962,.14022456165286365,-182.3481329994931,-627.6157264950939],
 [1685,1.7858474580520705,.16571673414706756,-203.5246087824123,-724.0471391042217],
 [1686,1.9265171401699313,.1960230861710924,-225.58190978669285,-835.1175415078651],
 [1687,2.1070030303125784,.23328962910773,-252.58169886952408,-968.4923392648411],
 [1688,2.3409769392279007,.28074672547573765,-281.08068586553793,-1128.2223349204598],
 [1689,2.6441468280924645,.34848963167157265,-294.9858155682961,-1316.042986373598],
 [1690,3.064323492216027,.3822620799097966,-327.93155215991754,-1488.795761538385],
]
OVERSCAN = [[1.471577890391003,.002367747087426085,-102.62794625253386],
            [-.002367747087426085,1.471577890391003,6.02822978238306]]

def smoother(t):
    t = float(np.clip(t,0,1))
    return t**3 * (10 - 15*t + 6*t*t)

def matrices():
    out = {}
    for n in range(1643,1661):
        dx,dy = OPEN_ENTRY[n-1643] if n<=1655 else (-13.19*(1660-n),-5.73*(1660-n))
        out[n] = np.array([[1,0,dx],[0,1,dy]],np.float64)
    for n,a,b,dx,dy in CLOSED:
        out[n] = np.array([[a,-b,dx],[b,a,dy]],np.float64)
    out[1661] = 2*out[1662] - out[1663]
    return out

def dump(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def source_path(cache, n):
    return cache / f'f{n:05}.png'

def ensure_sources(original, cache, frames):
    cache.mkdir(parents=True,exist_ok=True)
    missing = sorted(n for n in frames if not source_path(cache,n).exists())
    if not missing:
        return
    selected = '+'.join(f'eq(n\\,{n})' for n in missing)
    subprocess.run(['ffmpeg','-v','error','-y','-threads','2','-i',str(original),
        '-map','0:v:0','-vf','select='+selected,'-fps_mode','passthrough',
        '-compression_level','2','-start_number','0',str(cache/'temp_%05d.png')],check=True)
    for j,n in enumerate(missing):
        (cache/f'temp_{j:05}.png').replace(source_path(cache,n))

def calibrate(cache):
    outpath = HERE/'references/shot54_calibrated.json'
    motion = matrices()
    if outpath.exists():
        return {r['frame']:r for r in json.loads(outpath.read_text())}
    sources = {pose:np.asarray(Image.open(source_path(cache,n)).convert('RGB').resize(
        (480,270),Image.Resampling.LANCZOS),np.float32)
        for pose,n in [('open',1660),('closed',1667)]}
    records=[]
    for n,M in sorted(motion.items()):
        pose='open' if n<1661 else 'closed'
        target=np.asarray(Image.open(source_path(cache,n)).convert('RGB').resize(
            (480,270),Image.Resampling.LANCZOS),np.float32)
        scaled=M.copy();scaled[:,2]/=4
        warped=cv2.warpAffine(sources[pose],scaled,(480,270),flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE)
        mask=cv2.warpAffine(np.ones((270,480),np.uint8),scaled,(480,270))
        mask=cv2.erode(mask,np.ones((7,7),np.uint8));mask[:8]=0;mask[225:]=0
        mask[:,:8]=0;mask[:,-8:]=0;valid=mask[::2,::2]>0
        t=target[::2,::2][valid]
        def loss(sigma, return_exposure=False):
            im=cv2.GaussianBlur(warped,(0,0),max(.03,sigma/4))[::2,::2][valid]
            toward=255-im
            expo=float(np.clip(np.sum(toward*(t-im))/(np.sum(toward*toward)+1e-8),0,.96))
            error=float(np.mean(np.minimum((im*(1-expo)+255*expo-t)**2,2500)))
            return (expo,error) if return_exposure else error
        opt=minimize_scalar(loss,bounds=(.03,40 if pose=='open' else 15),
            method='bounded',options={'xatol':.05,'maxiter':28})
        expo,error=loss(opt.x,True)
        records.append(dict(frame=n,pose=pose,matrix=M.tolist(),
            blur_sigma=round(float(opt.x),4),exposure=round(expo,6),
            source_fit_rms=round(math.sqrt(error),4)))
    dump(outpath,records)
    dump(HERE/'references/shot54_character_motion.json',
         [dict(frame=n,matrix=M.tolist(),reference_frame=1660 if n<1661 else 1667)
          for n,M in sorted(motion.items())])
    dump(HERE/'references/shot54_overscan_alignment.json',dict(matrix=OVERSCAN,
         matches=332,inliers=247,median_error_pixels=2.048,
         method='SIFT similarity registration; approved portrait retained at native detail'))
    return {r['frame']:r for r in records}

class Shot54:
    def __init__(self,controls):
        self.controls=controls
        self.captions=Captions()
        self.keys={pose:np.asarray(Image.open(HERE/'assets'/f'a13_v17_{pose}_master.png')
            .convert('RGB').resize((W,H),Image.Resampling.LANCZOS))
            for pose in ('open','closed')}
        extra=np.asarray(Image.open(HERE/'assets/a13_v17_open_extended.png')
            .convert('RGB').resize((W,H),Image.Resampling.LANCZOS))
        T=np.array(OVERSCAN,np.float64)
        world=cv2.warpAffine(extra,T,(2880,1620),flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        valid=cv2.warpAffine(np.ones((H,W),np.float32),T,(2880,1620),flags=cv2.INTER_LINEAR)
        yy,xx=np.mgrid[:H,:W]
        ax=np.clip((W-xx)/100,0,1);ay=np.clip((H-yy)/110,0,1)
        alpha=(ax*ax*(3-2*ax)*ay*ay*(3-2*ay)).astype(np.float32)
        world[:H,:W]=self.keys['open']*alpha[:,:,None]+world[:H,:W]*(1-alpha[:,:,None])
        valid[:H,:W]=1
        self.keys['open']=np.clip(world,0,255).astype(np.uint8)
        self.valid_open=cv2.GaussianBlur(valid,(0,0),12)
        self.closed_matte=np.zeros((H,W),np.float32)
        polygon=np.array([(690,-100),(2050,-100),(2050,1250),(650,1250),
            (555,890),(500,680),(560,460),(650,270)],np.int32)
        cv2.fillPoly(self.closed_matte,[polygon],1)
        self.closed_matte=cv2.GaussianBlur(self.closed_matte,(0,0),20)
        self.gift=Image.open(HERE/'assets/a12_gift_00.png').convert('RGB')
        self.feet=camera(Image.open(HERE/'assets/a08_feet_00.png').convert('RGB'),1.012)
        self.kiss=None
        self.metrics=[];self.static_checks=[];self.kiss_metrics=[]
        self.first_feet=None

    def render(self,n,original):
        if 1304<=n<1342:
            out=self.captions.overlay(self.feet.copy(),n,original)
            region=np.asarray(out)[:900]
            if self.first_feet is None:self.first_feet=region.copy()
            delta=int(np.max(np.abs(region.astype(np.int16)-self.first_feet.astype(np.int16))))
            self.static_checks.append(dict(frame=n,region=[0,0,W,900],
                actual_pre_encode_max_pixel_delta=delta))
            return out,'static_feet'
        if 1418<=n<1442:
            if self.kiss is None:
                self.kiss=CelMotion('a09_kisswide',
                    [f'a09_kisswide_{k:02}.png' for k in (0,1,2,3,5,6,7,8)],HERE/'qa')
            cel=self.kiss.sample(n,1342,1496)
            p=smoother((n-1418)/12);settle=smoother((n-1430)/12)
            zoom=2.185+.030*settle
            wide=camera(cel,1.015,.495,.5)
            close=camera(cel,zoom,.34,.30)
            if p<1:
                close=Image.fromarray(cv2.GaussianBlur(np.asarray(close),(0,0),max(.1,4*(1-p))))
            im=Image.blend(wide,close,p)
            if 0<p<1:im=Image.blend(im,Image.new('RGB',(W,H),'white'),.07*math.sin(math.pi*p))
            self.kiss_metrics.append(dict(frame=n,close_dissolve=p,close_zoom=zoom))
            return self.captions.overlay(im,n,original),'soft_kiss_push'
        if 1638<=n<1643:
            im=camera(self.gift,1.024-.020*ease((n-1605)/37))
            return self.captions.overlay(im,n,original),'gift_cut'
        if 1691<=n<1710:
            return original.copy(),'original_optical_exit'
        control=self.controls[n];pose=control['pose']
        M=np.asarray(control['matrix'],np.float64)
        im=cv2.warpAffine(self.keys[pose],M,(W,H),flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE)
        sigma=control['blur_sigma'];exposure=control['exposure']
        if sigma>.06:im=cv2.GaussianBlur(im,(0,0),sigma)
        if exposure>0:im=np.clip(im.astype(np.float32)*(1-exposure)+255*exposure,0,255).astype(np.uint8)
        mask=self.valid_open if pose=='open' else self.closed_matte
        matte=cv2.warpAffine(mask,M,(W,H),flags=cv2.INTER_LINEAR)
        result=im.astype(np.float32)*matte[:,:,None]+np.asarray(original)*(1-matte[:,:,None])
        out=Image.fromarray(np.clip(result,0,255).astype(np.uint8))
        a,b=M[0,:2];c,d=M[1,:2]
        self.metrics.append(dict(frame=n,pose=pose,matrix=M.tolist(),
            zoom=float(math.hypot(a,c)),angle_degrees=float(math.degrees(math.atan2(c,a))),
            blur_sigma=sigma,exposure=exposure,
            axis_scale_difference=float(abs(math.hypot(a,c)-math.hypot(b,d))),
            axes_dot_product=float(a*b+c*d)))
        return self.captions.overlay(out,n,original),pose+'_portrait'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--original',required=True,type=Path)
    ap.add_argument('--base',required=True,type=Path,help='Validated V1.6 complete MP4')
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--qa-dir',type=Path,default=HERE/'qa/final_v17')
    ap.add_argument('--preview',action='store_true')
    ap.add_argument('--preview-window',choices=['smile','early'],default='smile')
    args=ap.parse_args()
    args.qa_dir.mkdir(parents=True,exist_ok=True);args.output.parent.mkdir(parents=True,exist_ok=True)
    cache=HERE/'qa/source_frames_v17'
    needed=set(n for _,a,b in PATCHES for n in range(a,b))
    ensure_sources(args.original,cache,needed)
    r=Shot54(calibrate(cache))
    start,stop=(1614,1722) if args.preview_window=='smile' else (1284,1482)
    if not args.preview:start,stop=0,COUNT
    select=f'select=between(n\\,{start}\\,{stop-1})'
    decoder_cmd=['ffmpeg','-v','error','-threads','2','-i',str(args.base),'-map','0:v:0']
    if args.preview:decoder_cmd+=['-vf',select]
    decoder_cmd+=['-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    decoder=subprocess.Popen(decoder_cmd,stdout=subprocess.PIPE,stderr=open(args.qa_dir/'decode.log','w'))
    silent=args.output.with_name(args.output.stem+'_video_only.mp4')
    encoder=subprocess.Popen(['ffmpeg','-v','warning','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an',
        '-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p',
        '-c:v','libx264','-preset','fast','-crf','16','-threads','4',
        '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',
        '-movflags','+faststart',str(silent)],stdin=subprocess.PIPE,
        stderr=open(args.qa_dir/'encode.log','w'))
    began=time.monotonic();counts=Counter()
    try:
        for n in range(start,stop):
            raw=read_exact(decoder.stdout,W*H*3)
            if len(raw)!=W*H*3:raise RuntimeError(f'Unexpected source end at frame {n}')
            if any(a<=n<b for _,a,b in PATCHES):
                original=Image.open(source_path(cache,n)).convert('RGB')
                out,label=r.render(n,original);raw=out.tobytes()
            else:
                label='retained_v16';out=None
            encoder.stdin.write(raw);counts[label]+=1
            if n in REVIEW:
                if out is None:out=Image.frombytes('RGB',(W,H),raw)
                out.resize((960,540),Image.Resampling.LANCZOS).save(args.qa_dir/f'pre_f{n:05}.jpg',quality=94)
            if n%300==0 or n==stop-1:
                print(json.dumps(dict(frame=n,stop=stop,seconds=round(time.monotonic()-began,1))),flush=True)
        if decoder.stdout.read(1):raise RuntimeError('Additional source frames')
        if decoder.wait()!=0:raise RuntimeError('Base decode failed')
        encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('Encode failed')
        mux=['ffmpeg','-v','error','-y','-i',str(silent)]
        if args.preview:mux+=['-ss',str(start/FPS),'-t',str((stop-start)/FPS)]
        mux+=['-i',str(args.original),'-map','0:v:0','-map','1:a:0','-c','copy',
              '-movflags','+faststart',str(args.output)]
        subprocess.run(mux,check=True);silent.unlink()
        dump(args.qa_dir/'trajectory.json',r.metrics)
        dump(args.qa_dir/'static_feet.json',r.static_checks)
        dump(args.qa_dir/'kiss_push.json',r.kiss_metrics)
        dump(args.qa_dir/'caption_states.json',r.captions.log)
        dump(args.qa_dir/'render_summary.json',dict(version='V1.7',start=start,stop=stop,
             frames=stop-start,fps=FPS,resolution=[W,H],patched_ranges=PATCHES,
             counts=dict(counts),elapsed_seconds=time.monotonic()-began,preview=args.preview))
        print(json.dumps(dict(status='complete',output=str(args.output),seconds=round(time.monotonic()-began,1))),flush=True)
    finally:
        for p in (decoder,encoder):
            if p.poll() is None:p.terminate()

if __name__=='__main__':main()
