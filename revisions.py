#!/usr/bin/env python3
"""Targeted V1.5 revision, keeping the existing edit and original frame clock."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import compose as base
from compose import cv2, np, Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
W, H, FPS, COUNT = base.W, base.H, base.FPS, base.COUNT
PATCHES = [
    ('restore_v14_title', 265, 343),
    ('cocoon_scene', 654, 724),
    ('intro_entry', 724, 813),
    ('static_silhouette', 983, 1063),
    ('smile_entry_exit', 1643, 1697),
    ('restore_v14_costume_cuts', 2027, 2062),
    ('stable_hand', 2218, 2272),
]


class RevisionRenderer(base.Renderer):
    def __init__(self, original, qa_dir):
        super().__init__(original, qa_dir)
        # The raised hand and adjoining wrist retain the exact master anatomy.
        motion = self.motion['a18_laugh']
        guard = base.feather_region(motion.size, (.75, .29, 1., 1.))
        iw, ih = motion.size
        guard[int(.33*ih):, int(.77*iw):] = 1.
        motion.mask *= 1-guard
        self.hand_guard = guard
        self.smile_cel = Image.open(HERE/'assets/a13_exit_cel.png').convert('RGBA')
        bounds = self.smile_cel.getchannel('A').getbbox()
        self.smile_cel = self.smile_cel.crop(bounds)
        # Reuse a clear portion of the painted pavement, preserving its light
        # and brushwork behind the now independently moving character cel.
        master = self.motion['a13_smile'].images[0]
        iw, ih = master.size
        self.smile_floor = master.crop((int(.02*iw),int(.07*ih),
                                      int(.39*iw),int(.59*ih))).resize(
                                      (W,H),Image.Resampling.BICUBIC)
        self.smile_floor = self.smile_floor.filter(ImageFilter.GaussianBlur(1.8))
        self.name_font = ImageFont.truetype(
            str(HERE/'assets/NotoSansCJKsc-Regular.otf'), 154)
        self.trajectory=[]
        self.props={}

    def intro(self, cel, t):
        # Original shot grammar: right-to-left flash, close portrait, pull-back,
        # then the character name appears in the space on the right.
        pull=base.ease((t-9)/17)
        close=base.camera(cel,2.48-1.47*pull,.265+.235*pull,.35+.15*pull)
        if t<10:
            enter=base.ease(t/9)
            dx=round(1040*(1-enter))
            im=Image.new('RGB',(W,H),(255,255,255))
            im.paste(close,(dx,0))
            xx=np.arange(W,dtype='float32')[None,:]
            # A soft optical boundary hides the image edge during the flash.
            reveal=np.clip((xx-dx)/160,0,1)
            edge_mask=np.repeat((reveal*255).astype('uint8'),H,axis=0)
            im=Image.composite(im,base.WHITE,Image.fromarray(edge_mask))
            edge=dx+80*(1-enter)
            streak=np.exp(-((xx-edge)/100)**2)*.55*math.sin(math.pi*enter)
            mask=np.repeat((streak*255).astype('uint8'),H,axis=0)
            im=Image.composite(base.WHITE,im,Image.fromarray(mask))
            im=base.expose(im,1-base.ease(t/6))
            im=base.blurred(im,3.5*(1-enter))
        else:
            im=close
        if t>=24:
            opacity=base.ease((t-24)/7)
            text=Image.new('RGBA',(W,H),(0,0,0,0));d=ImageDraw.Draw(text)
            for k,char in enumerate('流萤'):
                d.text((1590,236+k*190),char,font=self.name_font,
                       fill=(85,213,214,round(255*opacity)),stroke_width=2,
                       stroke_fill=(242,255,255,round(240*opacity)))
            out=im.convert('RGBA');out.alpha_composite(text);im=out.convert('RGB')
        self.trajectory.append({'scene':'intro','frame':724+t,
            'entry_dx':round(1040*(1-base.ease(t/9))) if t<10 else 0,
            'zoom':round(2.48-1.47*pull,5),'name_visible':t>=24})
        return im

    def smile(self, t, source):
        # All body crop edges remain below the frame during the pull-back.
        enter=base.ease(t/11)
        pull=base.ease((t-11)/17)
        leave=base.ease((t-33)/18)
        height=round(1400-260*pull)
        width=round(self.smile_cel.width*height/self.smile_cel.height)
        center=-1040+(2160*enter)+170*pull+2340*leave
        bottom=1265-65*pull
        xy=(round(center-width/2),round(bottom-height))
        layer=self.smile_cel.resize((width,height),Image.Resampling.LANCZOS)
        floor=base.camera(self.smile_floor,1.018-.012*base.ease(t/53))
        out=floor.convert('RGBA');out.alpha_composite(layer,xy)
        im=out.convert('RGB')
        if t<8:
            im=base.expose(im,.90*(1-base.ease(t/7)))
            im=base.blurred(im,4*(1-base.ease(t/7)))
        # The following shot is brought in on its original transition frames.
        if t>=49:im=Image.blend(im,source,base.ease((t-49)/4))
        self.trajectory.append({'scene':'smile','frame':1643+t,
             'center_x':round(center,3),'height':height,'bottom':bottom,
             'entry_progress':round(enter,5),'shrink_progress':round(pull,5),
             'exit_progress':round(leave,5)})
        return im

    def prop(self,t):
        if t not in self.props:
            self.props[t]=Image.open(HERE/'assets/prop_rotation'/f'{t:02}.png').convert('RGBA')
        return self.props[t]

    def render(self,n,source,original,current):
        if 265<=n<343:
            return source,'restore_v14_title'
        if 2027<=n<2062:
            return source,'restore_v14_costume_cuts'
        if 654<=n<724:
            t=n-654;cel=self.motion['a03_cocoon_bg'].sample(n,654,724)
            im=base.camera(cel).convert('RGBA');prop=self.prop(t)
            # Restrained bloom and a cold rim seat the cel-shaded object inside
            # the bright crystal space rather than above it like a product shot.
            glow=Image.new('RGBA',(W,H),(143,237,251,0))
            glow.putalpha(prop.getchannel('A').filter(ImageFilter.GaussianBlur(10)).point(lambda x:round(x*.17)))
            im.alpha_composite(glow);im.alpha_composite(prop)
            im=im.convert('RGB')
            im=base.expose(im,.25*(1-base.ease(t/4)))
            if n>=720:
                p=base.ease((n-720)/4)
                xx=np.arange(W,dtype='float32')[None,:]
                edge=W*(1-p)
                glint=np.exp(-((xx-edge)/150)**2)*.72
                mask=np.repeat((glint*255).astype('uint8'),H,axis=0)
                im=Image.composite(base.WHITE,im,Image.fromarray(mask))
            return self.captions.overlay(im,n,original),'cocoon_scene'
        if 724<=n<813:
            cel=self.motion['a04_intro'].sample(n,724,813)
            im=self.intro(cel,n-724)
            return self.captions.overlay(im,n,original),'intro_entry'
        if 983<=n<1063:
            im=base.camera(self.motion['a07_park'].images[0])
            return self.captions.overlay(im,n,original),'static_silhouette'
        if 1643<=n<1697:
            im=self.smile(n-1643,source)
            return self.captions.overlay(im,n,original),'smile_entry_exit'
        if 2218<=n<2272:
            cel=self.motion['a18_laugh'].sample(n,2218,2272)
            # Fixed framing keeps the hand stationary in screen coordinates;
            # hair outside the protected anatomy continues its existing breeze.
            im=base.camera(cel,1.02,.5,.5)
            t=n-2218
            im=base.blurred(im,6*(1-base.ease(t/9)))
            im=base.expose(im,.38*(1-base.ease(t/6)))
            return self.captions.overlay(im,n,original),'stable_hand'
        return current,'retained_v15'


REVIEW=sorted(set([264,265,270,300,336,342,343,
    654,660,675,690,705,719,720,721,723,724,726,729,732,735,738,
    743,748,754,765,792,812,813,982,983,1000,1020,1040,1062,1063,
    1642,1643,1646,1649,1652,1654,1661,1668,1670,1676,1680,1686,
    1689,1692,1696,1697,2026,2027,2033,2038,2040,2048,2053,2055,
    2061,2062,2217,2218,2224,2230,2240,2250,2260,2271,2272]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True,type=Path,help='V1.4 master')
    ap.add_argument('--original',required=True,type=Path)
    ap.add_argument('--base',required=True,type=Path,help='V1.5 finished video')
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--preview',action='store_true')
    ap.add_argument('--qa-dir',type=Path,default=HERE/'qa')
    args=ap.parse_args();args.qa_dir.mkdir(parents=True,exist_ok=True)
    r=RevisionRenderer(args.original,args.qa_dir)
    decoders=[];size=W*H*3
    select='+'.join(f'eq(n\\,{n})' for n in REVIEW)
    for p,label in [(args.source,'v14'),(args.original,'original'),(args.base,'v15')]:
        cmd=['ffmpeg','-v','error','-threads','2','-i',str(p),'-map','0:v:0']
        if args.preview:cmd+=['-vf',f'select={select}']
        cmd+=['-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
        decoders.append(subprocess.Popen(cmd,stdout=subprocess.PIPE,
            stderr=open(args.qa_dir/(label+'_decode.log'),'w')))
    video=args.output.with_name(args.output.stem+'_video_only.mp4')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not args.preview:
        encoder=subprocess.Popen(['ffmpeg','-v','warning','-y','-f','rawvideo',
            '-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an',
            '-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p',
            '-c:v','libx264','-preset','fast','-crf','17','-threads','4',
            '-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709',
            '-colorspace','bt709','-movflags','+faststart',str(video)],
            stdin=subprocess.PIPE,stderr=open(args.qa_dir/'encode.log','w'))
    started=time.monotonic();counts={};static_checks=[];prev=None;motion=[]
    try:
        for n in REVIEW if args.preview else range(COUNT):
            raw=[base.read_exact(p.stdout,size) for p in decoders]
            if any(len(x)!=size for x in raw):raise RuntimeError(f'Short source at {n}')
            source,original,current=[Image.frombytes('RGB',(W,H),x) for x in raw]
            im,label=r.render(n,source,original,current)
            counts[label]=counts.get(label,0)+1
            if not args.preview:encoder.stdin.write(im.tobytes())
            if n in REVIEW:
                im.resize((960,540),Image.Resampling.LANCZOS).save(
                    args.qa_dir/f'f{n:05d}.jpg',quality=94)
            # Compare actual pre-encode protected pixels, excluding lyrics.
            region=None
            if 983<=n<1063:region=(0,0,1920,900)
            if 2230<=n<2272:region=(1635,465,1920,900)
            if region:
                part=np.array(im.crop(region));key='park' if n<1063 else 'hand'
                if key not in r.before:r.before[key]=part
                error=int(np.max(np.abs(part.astype('int16')-r.before[key].astype('int16'))))
                static_checks.append({'frame':n,'region':key,'max_pixel_change':error})
            thumb=np.asarray(im.resize((160,90)).convert('L')).astype('float32')
            mad=0 if prev is None else float(np.abs(thumb-prev).mean())
            prev=thumb;motion.append({'frame':n,'scene':label,'mad':round(mad,5)})
            if n%180==0 or n==COUNT-1:
                print(json.dumps({'frame':n,'total':COUNT,'seconds':round(time.monotonic()-started,1)}),flush=True)
        for p in decoders:
            if p.stdout.read(1):raise RuntimeError('Unexpected additional source frames')
            if p.wait()!=0:raise RuntimeError('Source decode failed')
        if not args.preview:
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('Encode failed')
            subprocess.run(['ffmpeg','-v','error','-y','-i',str(video),'-i',str(args.original),
                '-map','0:v:0','-map','1:a:0','-c','copy','-movflags','+faststart',str(args.output)],check=True)
        (args.qa_dir/'trajectory.json').write_text(json.dumps(r.trajectory,indent=2))
        (args.qa_dir/'static_checks.json').write_text(json.dumps(static_checks,indent=2))
        (args.qa_dir/'caption_states.json').write_text(json.dumps(r.captions.log,ensure_ascii=False))
        (args.qa_dir/'motion_metrics.json').write_text(json.dumps(motion,indent=2))
        (args.qa_dir/'render_summary.json').write_text(json.dumps({'frames':COUNT,'fps':FPS,
            'resolution':[W,H],'patched_ranges':PATCHES,'counts':counts,
            'preview_only':args.preview,'elapsed_seconds':time.monotonic()-started},indent=2))
        print(json.dumps({'status':'complete','preview':args.preview,'seconds':round(time.monotonic()-started,1)}),flush=True)
    finally:
        for p in decoders:
            if p.poll() is None:p.terminate()
        if not args.preview and encoder.poll() is None:encoder.terminate()


if __name__=='__main__':main()
