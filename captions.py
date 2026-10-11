"""Recover original subtitle glyph pixels and their per-frame highlight state.

Typography is used as a search stencil only. Actual displayed glyph colors,
highlight, position and fade come from each original video frame.
"""
from pathlib import Path
import json
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
ROI=(900,920,1910,1030)
LINES=[
 ('eye',495,654,'一直注视着如同烈火燃烧美丽的瞳孔',[540,570,600,630]),
 ('cocoon',654,724,'逐渐混合在一起',[675,690,705]),
 ('intro',724,813,'却意外散发出光辉',[750,765,780,795]),
 ('duo',813,881,'自不用说或许之类的话',[825,840,855]),
 ('inner',881,973,'因为是我的内心',[900,915,930,945]),
 ('park',973,1064,'所以已经知晓了啊',[990,1005,1020,1035,1050]),
 ('run',1090,1304,'今日 我在融雪中奔跑',[1110,1140,1170,1215,1245,1290]),
 ('love',1304,1421,'今日 我喜欢上了你',[1320,1350,1365,1380,1395]),
 ('funny',1421,1482,'真可笑呢',[1440,1455,1470]),
 ('scenery',1482,1565,'在这什么都没有的风景中',[1500,1515,1530,1545]),
 ('gift',1565,1646,'在这一分一秒中',[1575,1590,1620,1635]),
 ('smile',1646,1700,'寻找着二人的契机',[1650,1665,1680]),
 ('snow',1950,2230,'雪 只是回想着纯白的回忆而已？',[1980,2025,2070,2115,2160,2190,2205]),
]

def hp(im):
 g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32)
 return np.abs(g-cv2.GaussianBlur(g,(0,0),3))

def text_template(text,size,gap,stroke):
 f=ImageFont.truetype(str(ROOT/'assets/NotoSansCJKsc-Regular.otf'),size)
 im=Image.new('L',(len(text)*(size+gap)+12,size*2),0);d=ImageDraw.Draw(im);x=5
 for c in text:
  if c==' ':x+=20;continue
  d.text((x,0),c,font=f,fill=255,stroke_width=stroke,stroke_fill=255);x+=size+gap
 b=im.getbbox();return np.array(im.crop((b[0]-3,b[1]-3,b[2]+3,b[3]+3)))

def glyph_stencil(signal,text,size,base_xy):
 chars=[c for c in text if c!=' '];font=ImageFont.truetype(str(ROOT/'assets/NotoSansCJKsc-Regular.otf'),size)
 glyphs=[]
 for c in chars:
  im=Image.new('L',(size+10,size+24),0);ImageDraw.Draw(im).text((5,0),c,font=font,fill=255,stroke_width=1,stroke_fill=255);glyphs.append(np.array(im))
 scores=[cv2.matchTemplate(signal,g.astype(np.float32),cv2.TM_CCORR_NORMED) for g in glyphs]
 best=(-1,None);N=scores[0].shape[1];expected=58.
 # Independent ordered glyph matching accommodates the original irregular spaces.
 for yy in range(scores[0].shape[0]):
  if not 4<=yy<=43:continue
  dp=scores[0][yy].copy();backs=[]
  for sc in scores[1:]:
   nxt=np.full(N,-100.,np.float32);arg=np.zeros(N,np.int16)
   for shift in range(37,101):
    vals=dp[:-shift]-.00030*(shift-expected)**2
    update=vals>nxt[shift:];nxt[shift:][update]=vals[update];arg[shift:][update]=np.arange(N-shift)[update]
   dp=nxt+sc[yy];backs.append(arg)
  ends=np.arange(N);# The original line ends in the final 120 pixels of ROI.
  dp[ends<865]=-100;dp[ends>970]=-100
  if len(chars)==7 and '逐渐' in text:dp[ends>920]=-100;dp[ends<810]=-100
  x=int(dp.argmax());value=float(dp[x])/len(chars)
  if value>best[0]:
   xs=[x]
   for arg in reversed(backs):x=int(arg[x]);xs.append(x)
   best=(value,(yy,list(reversed(xs))))
 value,(yy,xs)=best;stencil=np.zeros(signal.shape,np.uint8)
 for x,g in zip(xs,glyphs):stencil[yy:yy+g.shape[0],x:x+g.shape[1]]=np.maximum(stencil[yy:yy+g.shape[0],x:x+g.shape[1]],g)
 return stencil,dict(score=value,draw_y=yy+920,xs=[x+900 for x in xs])

def fit():
 out=ROOT/'assets/caption_masks';out.mkdir(exist_ok=True)
 meta=[];sheet=Image.new('RGB',(1010,len(LINES)*120),'#101010');d=ImageDraw.Draw(sheet)
 for j,(name,a,b,text,samples) in enumerate(LINES):
  ims=[np.array(Image.open(ROOT/'references/lyrics'/f'f{n:05}.png').crop((900,0,1910,110))) for n in samples]
  maps=np.stack([hp(im) for im in ims]);signal=np.percentile(maps,45,axis=0).astype(np.float32)
  # The source glyphs are fixed above y=1003. Reject unrelated bottom edges.
  signal[88:]=0
  best=(-1,None)
  for sz in range(37,44):
   for gap in range(12,23):
    for stroke in (0,1):
     tem=text_template(text,sz,gap,stroke)
     if tem.shape[1]>1010:continue
     score=cv2.matchTemplate(signal,tem.astype(np.float32),cv2.TM_CCORR_NORMED)
     _,v,_,xy=cv2.minMaxLoc(score)
     if v>best[0]:best=(v,(sz,gap,stroke,xy,tem))
  score,(sz,gap,stroke,(x,y),tem)=best
  stencil,glyph_info=glyph_stencil(signal,text,40 if name!='eye' else 39,[x,y])
  permit=cv2.dilate((stencil>0).astype(np.uint8),np.ones((7,7),np.uint8))
  consensus=(signal>7).astype(np.uint8)*permit
  consensus=cv2.morphologyEx(consensus,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))*permit
  mask=cv2.dilate(consensus,np.ones((3,3),np.uint8))*permit
  # Exact source strokes only; no background rectangle is carried over.
  Image.fromarray(mask*255).save(out/f'{name}.png')
  layer=extract(ims[len(ims)//2],mask)
  sheet.paste(Image.fromarray(layer,'RGBA'),(0,j*120+18),Image.fromarray(layer[:,:,3]))
  d.text((5,j*120),f'{name}: {a}-{b} fit={score:.3f} font={sz}/{gap}/{stroke} pos={x+900},{y+920}',fill='#ffa050')
  meta.append(dict(name=name,start=a,end=b,text=text,samples=samples,font_size=sz,gap=gap,stroke=stroke,xy=[x+900,y+920],fit_score=score,glyph_fit=glyph_info,mask=f'assets/caption_masks/{name}.png'))
  print(name,round(score,3),sz,gap,stroke,(x+900,y+920),flush=True)
 (ROOT/'captions.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 sheet.save(ROOT/'qa/caption_layers.jpg')

def extract(roi,mask):
 # Mask-based background estimation separates the text from its old artwork.
 bg=cv2.inpaint(roi,mask*255,3,cv2.INPAINT_TELEA)
 delta=np.max(np.abs(roi.astype(np.float32)-bg.astype(np.float32)),axis=2)
 alpha=np.clip((delta-4)/10,0,1)*mask
 result=np.dstack((roi,(alpha*255).astype(np.uint8)))
 return result

class Captions:
 def __init__(self):
  self.meta=json.loads((ROOT/'captions.json').read_text())
  self.masks={m['name']:np.array(Image.open(ROOT/m['mask']))//255 for m in self.meta}
  self.glyphs={};self.log=[]
  for m in self.meta:
   info=m['glyph_fit'];xs=info['xs'][:]
   if m['name']=='cocoon':xs[-1]=1748
   if m['name']=='love':xs[:2]=[1321,1377]
   size=39 if m['name']=='eye' else 40;f=ImageFont.truetype(str(ROOT/'assets/NotoSansCJKsc-Regular.otf'),size)
   glyphs=[]
   for char,x in zip([c for c in m['text'] if c!=' '],xs):
    canvas=Image.new('L',(size+10,size+24),0);d=ImageDraw.Draw(canvas);d.text((5,0),char,font=f,fill=255,stroke_width=1,stroke_fill=255)
    outer=np.array(canvas)/255.;inner=cv2.erode((outer>.5).astype('uint8'),np.ones((3,3),np.uint8));outline=np.clip((outer>.05).astype('uint8')-cv2.erode((outer>.05).astype('uint8'),np.ones((3,3),np.uint8)),0,1)
    glyphs.append(dict(char=char,x=x,y=info['draw_y'],outer=outer,inner=inner,outline=outline))
   self.glyphs[m['name']]=glyphs
 def overlay(self,frame,n,source):
  layers=[m for m in self.meta if m['start']<=n<m['end']]
  if not layers:return frame
  roi=np.array(source.crop(ROI));dst=frame.convert('RGBA')
  for m in layers:
   # Re-typeset complete clean glyphs; sample original frame for presence and fill.
   # This avoids carrying original character/background pixels into the new CG.
   g=cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY).astype(np.float32);high=np.abs(g-cv2.GaussianBlur(g,(0,0),3));mask=self.masks[m['name']]
   ink=float(np.mean(high[mask>0]));outside=cv2.dilate(mask,np.ones((11,11),np.uint8))-mask;noise=float(np.mean(high[outside>0]))
   opacity=float(np.clip((ink-noise-1.5)/9,0,1));states=[]
   if opacity<.02:continue
   for gg in self.glyphs[m['name']]:
    x,y=gg['x']-ROI[0],gg['y']-ROI[1];hh,ww=gg['outer'].shape;s=roi[y:y+hh,x:x+ww];sl=g[y:y+hh,x:x+ww]
    bg=cv2.inpaint(s,(gg['outer']>.05).astype('uint8')*255,3,cv2.INPAINT_TELEA);bl=cv2.cvtColor(bg,cv2.COLOR_RGB2GRAY).astype(np.float32)
    inner=gg['inner']>0;ratio=float(np.mean((sl-bl)[inner]>12)) if inner.any() else 0.
    fill=float(np.clip((ratio-.20)/.34,0,1));states.append(round(fill,4))
    a=np.maximum(gg['outer']*fill,gg['outline']*.9)*opacity
    rgba=np.zeros((hh,ww,4),np.uint8);rgba[:,:,:3]=250;rgba[:,:,3]=(a*255).astype('uint8');layer=Image.fromarray(rgba,'RGBA')
    # Cyan edges remain readable on the high-key crystal shot; dark-scene
    # typography keeps a restrained blue shadow. Highlight still comes from n.
    bright=float(np.median(roi))>170
    edge_color=(60,166,191,0) if bright else (45,74,99,0)
    edge_alpha=185 if bright else 80
    shadow=Image.new('RGBA',(ww,hh),edge_color);shadow.putalpha(Image.fromarray((gg['outline']*edge_alpha*opacity).astype('uint8')));dst.alpha_composite(shadow,(gg['x']+1,gg['y']+1));dst.alpha_composite(layer,(gg['x'],gg['y']))
   self.log.append(dict(frame=n,line=m['name'],opacity=round(opacity,4),fills=states))
  return dst.convert('RGB')

if __name__=='__main__':fit()
