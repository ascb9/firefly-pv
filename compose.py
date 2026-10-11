#!/usr/bin/env python3
"""Frame-locked Firefly PV: recovered illustrated cels, bounded inbetweens.

All animation keys are full-canvas GPT Image illustrations. Small-displacement
bidirectional interpolation only joins those keys; camera transforms are rigid.
The original running/title shot is copied at its original frame indices.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError:
    # Reuse the restored runtime during this session. Portable projects instead
    # install requirements.txt; no bundled executable dependency is required.
    recovered = Path(__file__).resolve().parent / 'deps'
    if recovered.is_dir():
        sys.path.insert(0, str(recovered))
    import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from captions import Captions, ROI

HERE = Path(__file__).resolve().parent
W, H, FPS, COUNT = 1920, 1080, 30, 2764
cv2.setNumThreads(2)
WHITE = Image.new('RGB', (W, H), 'white')
RUN_RANGE = (1110, 1304)
RANGES = [
    ('a01_start', 265, 343), ('a02_eye', 508, 654),
    ('a03_cocoon_bg', 654, 724), ('a04_intro', 724, 813),
    ('a05_duo', 813, 880), ('a06_duoclose', 880, 946),
    ('a07_park', 983, 1063), ('a08_feet', 1304, 1342),
    ('a09_kisswide', 1342, 1496), ('a11_group', 1545, 1605),
    ('a12_gift', 1605, 1643), ('a13_smile', 1643, 1697),
    ('a14_warm', 1950, 2027), ('a15_snowduo', 2027, 2062),
    ('a16_bright', 2062, 2116), ('a17_pendant', 2116, 2218),
    ('a18_laugh', 2218, 2272), ('a19_end', 2592, COUNT),
]


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def read_exact(stream, size):
    chunks = []
    while size:
        b = stream.read(size)
        if not b:
            break
        chunks.append(b)
        size -= len(b)
    return b''.join(chunks)


def camera(im, zoom=1., cx=.5, cy=.5):
    iw, ih = im.size
    cw = min(iw, ih * W / H) / max(1., zoom)
    ch = cw * H / W
    x = float(np.clip(cx * iw - cw / 2, 0, iw - cw))
    y = float(np.clip(cy * ih - ch / 2, 0, ih - ch))
    assert 0 <= x and x + cw <= iw + 1e-4
    assert 0 <= y and y + ch <= ih + 1e-4
    return im.transform((W, H), Image.Transform.EXTENT,
                        (x, y, x + cw, y + ch),
                        resample=Image.Resampling.BICUBIC)


def expose(im, amount):
    return Image.blend(im, WHITE, float(np.clip(amount, 0, 1)))


def blurred(im, radius):
    return im.filter(ImageFilter.GaussianBlur(radius)) if radius > .1 else im


def feather_region(size, rect):
    w, h = size
    m = Image.new('L', size, 0)
    ImageDraw.Draw(m).rectangle(tuple(int(v * (w if i % 2 == 0 else h))
                                    for i, v in enumerate(rect)), fill=255)
    return np.asarray(m.filter(ImageFilter.GaussianBlur(max(10, w * .025))),
                      np.float32) / 255.


def feather_ellipse(size, bounds):
    w, h = size
    m = Image.new('L', size, 0)
    ImageDraw.Draw(m).ellipse(tuple(int(v*(w if i%2==0 else h))
                                  for i,v in enumerate(bounds)),fill=255)
    return np.asarray(m.filter(ImageFilter.GaussianBlur(w*.018)),np.float32)/255.


class CelMotion:
    """Join registered full scenes using only small, measured drawn changes."""
    def __init__(self, name, files, qa_dir):
        self.name = name
        self.files = files
        self.images = [Image.open(HERE / 'assets' / f).convert('RGB') for f in files]
        self.size = self.images[0].size
        self.arrays = [np.asarray(im.resize(self.size, Image.Resampling.LANCZOS))
                       for im in self.images]
        w, h = self.size
        self.mask = np.ones((h, w), np.float32)
        regions = {
            'a01_start': (.27, .03, .86, .95),
            'a03_cocoon_bg': (.12, .18, .42, .79),
            'a05_duo': (.38, .0, 1., 1.),
            'a07_park': (.27, .40, .55, .98),
        }
        if name in regions:
            self.mask = feather_region(self.size, regions[name])
        if name == 'a02_eye':
            # Preserve the fully open eye, its iris and pupil, plus crystals.
            # Only the silver fringe above the eye and the far-left strands move.
            self.mask = feather_region(self.size, (0., 0., .60, .30))
            left = feather_region(self.size, (0., 0., .055, .95))
            self.mask = np.maximum(self.mask, left)
        if name == 'a12_gift':
            # Stable box, hands and rigid chest jewelry; hair/ribbon motion remains.
            self.mask = np.maximum(feather_region(self.size, (0., 0., .35, .24)),
                                   feather_region(self.size, (.78, 0., 1., .60)))
        # Lock face anatomy while allowing the surrounding drawn hair, ribbons
        # and clothing to move. These are broad, feathered stabilization areas.
        faces = {
            'a04_intro': (.10,.17,.43,.59),
            'a05_duo': (.63,.25,.84,.65),
            'a06_duoclose': (.28,.16,.73,.78),
            'a09_kisswide': (.07,.025,.47,.46),
            'a13_smile': (.65,.02,1.,.54),
            'a14_warm': (.48,.15,.79,.68),
            'a15_snowduo': (.53,.15,.80,.56),
            'a16_bright': (.32,.17,.74,.74),
            'a17_pendant': (.40,.04,.77,.51),
            'a18_laugh': (.39,.19,.86,.75),
            'a19_end': (.22,.14,.60,.61),
        }
        if name in faces:
            self.mask *= 1-feather_ellipse(self.size,faces[name])
        self.xx, self.yy = np.meshgrid(np.arange(w, dtype=np.float32),
                                      np.arange(h, dtype=np.float32))
        self.flows = {}
        self.metrics = []
        self.cache = qa_dir / 'flows'
        self.cache.mkdir(exist_ok=True)
        # Three breeze keys return through the registered master; no jump on wrap.
        self.sequence = list(range(len(files))) if name == 'a09_kisswide' else \
                        ([0, 1, 0] if len(files) == 2 else
                         [0, 1, 0, 2, 0] if len(files) == 3 else [0, 1, 0, 2, 3, 0])

    def flow(self, a, b):
        key = (a, b)
        if key in self.flows:
            return self.flows[key]
        file = self.cache / f'{self.name}_{a:02}_{b:02}.npz'
        if file.exists():
            z = np.load(file)
            f0, f1 = z['forward'], z['backward']
        else:
            w, h = self.size
            sw = 640
            sh = round(h * sw / w)
            A, B = [cv2.cvtColor(cv2.resize(self.arrays[k], (sw, sh)),
                                cv2.COLOR_RGB2GRAY) for k in (a, b)]
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            dis.setUseSpatialPropagation(True)
            f0 = dis.calc(A, B, None)
            f1 = dis.calc(B, A, None)
            for f in (f0, f1):
                f *= w / sw
            f0 = cv2.resize(f0, (w, h))
            f1 = cv2.resize(f1, (w, h))
            # At 1080p the maximum interpolated motion is 24 px per key pair.
            # Large redraw disagreements stay registered rather than bending faces.
            bound = 24. * w / W
            for f in (f0, f1):
                length = np.linalg.norm(f, axis=2)
                f *= np.minimum(1., bound / (length + 1e-5))[:, :, None]
                f *= self.mask[:, :, None]
            np.savez_compressed(file, forward=f0.astype('float16'),
                                backward=f1.astype('float16'))
        f0, f1 = f0.astype('float32'), f1.astype('float32')
        self.flows[key] = (f0, f1)
        mag = np.linalg.norm(f0, axis=2)
        self.metrics.append({'pair': [a, b], 'max_pixels_1080p': float(mag.max()*W/self.size[0]),
                             'p95_pixels_1080p': float(np.percentile(mag, 95)*W/self.size[0])})
        return self.flows[key]

    def sample(self, n, start, end):
        if self.name == 'a09_kisswide':
            position = np.clip((n-start) / (end-start-1), 0, 1) * (len(self.sequence)-1)
        else:
            cycle = 98 if self.name == 'a02_eye' else 84 if self.name == 'a19_end' else 76
            position = ((n-start) % cycle) / cycle * (len(self.sequence)-1)
        k = min(int(position), len(self.sequence)-2)
        q = ease(position-k)
        a, b = self.sequence[k], self.sequence[k+1]
        if q < 1e-5:
            # Keep unchanged background / protected anatomy exactly registered.
            im = self.arrays[a].astype('float32')
        else:
            f0, f1 = self.flow(a, b)
            wa = cv2.remap(self.arrays[a], self.xx-q*f0[:,:,0], self.yy-q*f0[:,:,1],
                           cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
            wb = cv2.remap(self.arrays[b], self.xx-(1-q)*f1[:,:,0], self.yy-(1-q)*f1[:,:,1],
                           cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
            im = wa.astype('float32')*(1-q) + wb.astype('float32')*q
        im = im*self.mask[:,:,None] + self.arrays[0]*(1-self.mask[:,:,None])
        return Image.fromarray(np.clip(im, 0, 255).astype('uint8'))


class Renderer:
    def __init__(self, original, qa_dir):
        self.qa_dir = qa_dir
        self.captions = Captions()
        self.motion = {}
        for name, _, _ in RANGES:
            files = sorted(p.name for p in (HERE/'assets').glob(name+'_*.png'))
            if name == 'a02_eye':
                files = ['a02_eye_00.png', 'a02_eye_01.png', 'a02_eye_hair02.png']
            if name == 'a09_kisswide':
                files = ['a09_kisswide_00.png', 'a09_kisswide_01.png',
                         'a09_kisswide_02.png', 'a09_kisswide_03.png',
                         'a09_kisswide_05.png', 'a09_kisswide_06.png',
                         'a09_kisswide_07.png', 'a09_kisswide_08.png']
            self.motion[name] = CelMotion(name, files, qa_dir)
        self.logo = Image.open(HERE/'assets/firefly_logo.png').convert('RGBA')
        bbox = self.logo.getchannel('A').point(lambda x:255 if x>80 else 0).getbbox()
        self.logo = self.logo.crop(bbox)
        self.before = {}
        self.bg = self.extract_frame(original, 1496)
        # Remove future subtitle only from the common crystalline exit plate.
        roi = np.array(self.bg.crop(ROI))
        meta = next(x for x in self.captions.meta if x['name']=='scenery')
        mask = self.captions.masks[meta['name']]*255
        roi = cv2.inpaint(roi, mask, 3, cv2.INPAINT_TELEA)
        self.bg.paste(Image.fromarray(roi), ROI[:2])
        yy, xx = np.mgrid[0:H, 0:W].astype('float32')
        self.wipe_axis = (xx/W + .15*yy/H) / 1.15
        self.shard_rank = np.zeros((H, W), np.float32)
        rng = np.random.default_rng(27)
        for y in range(-140, H+140, 190):
            for x in range(-170, W+170, 250):
                value = float(np.clip((x/W + .5*y/H)/1.5+rng.uniform(-.20,.20),0,1))
                pts = np.array([(x-90,y),(x+170,y-50),(x+230,y+170),(x+20,y+235)], np.int32)
                cv2.fillConvexPoly(self.shard_rank, pts, value)
        self.validation = set([264,265,276,300,330,342,343,507,508,513,525,570,615,653,
            654,660,675,690,705,718,723,724,737,765,807,812,813,818,828,845,870,879,
            880,896,925,945,946,982,983,1005,1045,1062,1063,1109,1110,1142,1200,
            1259,1260,1290,1303,1304,1320,1341,1342,1350,1380,1410,1418,1421,1428,
            1435,1440,1450,1470,1481,1482,1486,1490,1495,1496,1544,1545,1555,
            1578,1604,1605,1615,1642,1643,1660,1661,1680,1696,1697,1949,1950,
            1960,1980,2026,2027,2040,2061,2062,2070,2100,2115,2116,2125,2138,
            2139,2145,2180,2217,2218,2225,2260,2271,2272,2591,2592,2600,2620,
            2660,2700,2730,2750,2763])
        self.counts = {}

    @staticmethod
    def extract_frame(src, n):
        data = subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(src),
            '-vf',f'select=eq(n\\,{n})','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
        return Image.frombytes('RGB', (W,H), data)

    def add_logo(self, im, width, xy, opacity=1.):
        logo = self.logo.resize((width, round(self.logo.height*width/self.logo.width)),
                                Image.Resampling.LANCZOS)
        if opacity < 1:
            logo.putalpha(logo.getchannel('A').point(lambda x:round(x*opacity)))
        out = im.convert('RGBA')
        out.alpha_composite(logo, xy)
        return out.convert('RGB')

    def wipe(self, im, before, t):
        p = ease(t)
        m = np.clip((p*1.2-self.wipe_axis)/.2, 0, 1)
        out = Image.composite(im, before, Image.fromarray((m*255).astype('uint8')))
        glint = np.exp(-((self.wipe_axis-p)/.045)**2)*.34*math.sin(math.pi*p)
        return Image.composite(WHITE, out, Image.fromarray((glint*255).astype('uint8')))

    def render(self, n, source, original):
        if n == 1544:
            self.before[n] = source.copy()
        if RUN_RANGE[0] <= n < RUN_RANGE[1]:
            return original, 'original_running'
        match = next(((name,a,b) for name,a,b in RANGES if a<=n<b), None)
        if match is None:
            return source, 'retained_v14'
        name,a,b = match
        t, u = n-a, ease((n-a)/(b-a-1))
        cel = self.motion[name].sample(n, a, b)
        if name == 'a01_start':
            im = camera(cel, 1.025-.014*u, .501+.002*u, .500)
            im = self.add_logo(im, 1000, (400,590))
            im = expose(im, 1-ease(t/9))
            if n>=336: im = expose(im, ease((n-336)/6))
        elif name == 'a02_eye':
            settle = ease(t/32)
            im = camera(cel, 1.10-.10*settle, .53-.03*settle, .48+.02*settle)
            im = blurred(im, 5*(1-ease(t/15)))
            im = expose(im, .84*(1-ease(t/6)))
        elif name == 'a03_cocoon_bg':
            # Camera stays still: no invented shaking of the tiny figure.
            im = camera(cel)
            prop = Image.open(HERE/'assets/prop_rotation'/f'{t:02}.png').convert('RGBA')
            # Icy scene light blends the native rotating device with its backdrop.
            rgb = np.array(prop)
            rgb[:,:,:3] = np.clip(rgb[:,:,:3].astype('float32')*.88+
                                  np.array([216,249,255])*.12,0,255).astype('uint8')
            prop = Image.fromarray(rgb)
            im = im.convert('RGBA'); im.alpha_composite(prop); im = im.convert('RGB')
            im = expose(im, .25*(1-ease(t/4)))
            if n>=719: im=expose(im,.85*ease((n-719)/5))
        elif name == 'a04_intro':
            im = camera(cel, 1.008+.008*u, .5+.003*u, .5)
            im = expose(im, .98*(1-ease(t/14)))
            # Retain the already established character introduction label.
            if t>=14:
                d=ImageDraw.Draw(im)
                font=ImageFont.truetype(str(HERE/'assets/NotoSansCJKsc-Regular.otf'),118)
                for k,c in enumerate('流萤'):
                    d.text((1630,202+k*168),c,font=font,fill=(121,229,223),
                           stroke_width=2,stroke_fill=(239,255,255))
        elif name == 'a05_duo':
            im = camera(cel, 1.008+.012*u)
            if t<23: im=self.wipe(im,self.before[812],t/23)
            if n>=871:
                # Original foreground ochre sweep, before the next close shot.
                im=Image.blend(im,source,ease((n-871)/9))
        elif name == 'a06_duoclose':
            im=camera(cel,1.018+.014*u,.50+.005*u,.5)
            im=expose(im,.62*(1-ease(t/10)))
            if n>=935:im=Image.blend(im,source,ease((n-935)/11))
        elif name == 'a07_park':
            im=camera(cel,1.025-.018*u,.5,.5)
            im=blurred(im,4.5*(1-ease(t/9)))
        elif name == 'a08_feet':
            im=camera(cel,1.007+.010*u,.5+.001*u,.5)
            im=expose(im,.94*(1-ease(t/7)))
        elif name == 'a09_kisswide':
            # One common illustration, also during wide-to-close camera movement.
            push=ease((n-1418)/23)
            im=camera(cel,1.015+1.20*push,.495-.155*push,.50-.20*push)
            im=expose(im,.35*(1-ease(t/6)))
            if 1418<=n<1442:
                im=blurred(im,2.3*math.sin(math.pi*push))
                im=expose(im,.11*math.sin(math.pi*push))
            if n>=1482:
                q=ease((n-1482)/13)
                m=np.clip((q*1.3-self.shard_rank)/.24,0,1)
                im=Image.composite(self.bg,im,Image.fromarray((m*255).astype('uint8')))
                im=expose(im,.16*math.sin(math.pi*q))
        elif name == 'a11_group':
            im=camera(cel,1.015-.010*u)
            im=expose(im,.93*(1-ease(t/18)))
        elif name == 'a12_gift':
            im=camera(cel,1.024-.020*u)
            im=blurred(im,6*(1-ease(t/9)))
            im=expose(im,1-ease(t/9))
            if n>=1638:im=expose(im,ease((n-1638)/5))
        elif name == 'a13_smile':
            im=camera(cel,1.27-.25*ease(t/30),.60-.095*ease(t/30),.48+.02*u)
            im=blurred(im,6*(1-ease(t/13)))
            im=expose(im,.88*(1-ease(t/8)))
            if n>=1689:im=Image.blend(im,source,ease((n-1689)/8))
        elif name == 'a14_warm':
            im=camera(cel,1.40-.37*u,.53-.03*u,.47+.03*u)
            im=blurred(im,9*(1-ease(t/36)))
        elif name == 'a15_snowduo':
            im=camera(cel,1.012+.008*u)
            im=expose(im,.97*(1-ease(t/8)))
        elif name == 'a16_bright':
            im=camera(cel,1.25-.22*ease(t/28),.5,.5)
            im=blurred(im,5*(1-ease(t/11)))
        elif name == 'a17_pendant':
            pull=ease((n-2139)/11)
            im=camera(cel,2.50-1.48*pull,.39+.11*pull,.72-.22*pull)
            if n<2139:im=blurred(im,2.5*(1-ease(t/7)))
        elif name == 'a18_laugh':
            im=camera(cel,1.28-.26*ease(t/15),.5,.5)
            im=blurred(im,6*(1-ease(t/9)))
            im=expose(im,.38*(1-ease(t/6)))
        elif name == 'a19_end':
            im=camera(cel,1.047,.5,.507-.016*ease(t/112))
            im=self.add_logo(im,790,(1060,630))
            im=expose(im,.82*(1-ease(t/9)))
            if n>=2730:im=expose(im,ease((n-2730)/33))
        else:
            raise AssertionError(name)
        if n == 812:
            self.before[812] = im.copy()
        im=self.captions.overlay(im,n,original)
        return im,name


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True,type=Path,help='V1.4 visual master')
    ap.add_argument('--original',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--qa-dir',type=Path,default=HERE/'qa')
    ap.add_argument('--preview',action='store_true',help='Selected final frames only')
    args=ap.parse_args();args.qa_dir.mkdir(parents=True,exist_ok=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    r=Renderer(args.original,args.qa_dir)
    size=W*H*3
    decoders=[]
    for src,label in [(args.source,'source'),(args.original,'original')]:
        p=subprocess.Popen(['ffmpeg','-v','error','-threads','2','-i',str(src),
             '-map','0:v:0','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],
             stdout=subprocess.PIPE,stderr=open(args.qa_dir/(label+'_decode.log'),'w'))
        decoders.append(p)
    video=args.output.with_name(args.output.stem+'_video_only.mp4')
    if not args.preview:
        encoder=subprocess.Popen(['ffmpeg','-v','warning','-y','-f','rawvideo',
             '-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an',
             '-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p',
             '-c:v','libx264','-preset','fast','-crf','18','-threads','4',
             '-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709',
             '-colorspace','bt709','-movflags','+faststart',str(video)],
             stdin=subprocess.PIPE,stderr=open(args.qa_dir/'encode.log','w'))
    started=time.monotonic();prev=None;motion=[];sources={}
    try:
        for n in range(COUNT):
            raw=[read_exact(p.stdout,size) for p in decoders]
            if any(len(b)!=size for b in raw):raise RuntimeError(f'Short source at {n}')
            source,orig=[Image.frombytes('RGB',(W,H),b) for b in raw]
            if args.preview and n not in r.validation and n not in (812,1544):continue
            im,label=r.render(n,source,orig)
            sources[label]=sources.get(label,0)+1
            if not args.preview:encoder.stdin.write(im.tobytes())
            thumb=np.asarray(im.resize((160,90)).convert('L')).astype('float32')
            mad=0. if prev is None else float(np.abs(thumb-prev).mean())
            prev=thumb;motion.append({'frame':n,'scene':label,'mad':round(mad,5),
                                     'luma':round(float(thumb.mean()),3)})
            if n in r.validation:
                im.resize((960,540),Image.Resampling.LANCZOS).save(
                    args.qa_dir/f'f{n:05d}.jpg',quality=94)
            if n%180==0 or n==COUNT-1:
                print(json.dumps({'frame':n,'total':COUNT,'seconds':round(time.monotonic()-started,1)}),flush=True)
        for p in decoders:
            if p.stdout.read(1):raise RuntimeError('More frames than original timeline')
            if p.wait()!=0:raise RuntimeError('Decode failed')
        if not args.preview:
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('Encode failed')
            subprocess.run(['ffmpeg','-v','error','-y','-i',str(video),'-i',str(args.original),
                '-map','0:v:0','-map','1:a:0','-c','copy','-movflags','+faststart',str(args.output)],check=True)
        stats={name:{'assets':m.files,'motion_pairs':m.metrics} for name,m in r.motion.items()}
        (args.qa_dir/'animation_metrics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
        (args.qa_dir/'motion_metrics.json').write_text(json.dumps(motion,indent=2))
        (args.qa_dir/'caption_states.json').write_text(json.dumps(r.captions.log,ensure_ascii=False))
        (args.qa_dir/'render_summary.json').write_text(json.dumps({'frames':COUNT,'fps':FPS,
            'resolution':[W,H],'running_original_frames':list(RUN_RANGE),'scene_counts':sources,
            'elapsed_seconds':time.monotonic()-started,'preview_only':args.preview},indent=2))
    except Exception:
        for p in decoders:p.kill()
        if not args.preview:encoder.kill()
        raise


if __name__=='__main__':main()
