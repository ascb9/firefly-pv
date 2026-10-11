"""Native extruded textured SAM Cocoon rendering, including body/wing sides.
Requires numpy, Pillow, opencv-python-headless and moderngl; EGL software GPU is supported.
"""
from pathlib import Path
import math,json
import numpy as np,cv2,moderngl
from PIL import Image
R=Path(__file__).resolve().parent;W,H=1920,1080
VERT='''#version 330
in vec3 pos;in vec2 uv;in vec3 normal;in vec3 color;
uniform mat3 rotation;uniform vec2 center;uniform float scale;
out vec2 vuv;out vec3 vn;out vec3 vc;
void main(){vec3 p=rotation*pos;float perspective=1.0/(1.0-p.z*.18);
 gl_Position=vec4(center+vec2(p.x*scale/960.,p.y*scale/540.)*perspective,-p.z*.8,1.);
 vuv=uv;vn=rotation*normal;vc=color;}
'''
FRAG='''#version 330
uniform sampler2D art;uniform int is_side;in vec2 vuv;in vec3 vn;in vec3 vc;out vec4 outColor;
void main(){vec4 t=is_side==1?vec4(vc,1.):texture(art,vuv);if(t.a<.035)discard;
 vec3 n=normalize(vn);
 // Broad anime shading replaces the former product-like metallic glint.
 float nd=max(dot(n,normalize(vec3(-.4,.65,1.))),0.);
 float light=nd<.30?.91:(nd<.68?.965:1.025);
 float y=dot(t.rgb,vec3(.299,.587,.114));
 vec3 desat=mix(t.rgb,vec3(y)*vec3(.93,1.02,1.07),.34);
 vec3 icy=desat*.70+vec3(.69,.94,.98)*.30;
 // Shadows receive the same cold fill as the surrounding crystalline shot.
 icy=mix(icy,vec3(.40,.68,.75),max(0.,.32-y)*.30);
 float rim=pow(1.-abs(n.z),3.)*.12;
 vec3 rgb=clamp(icy*light+vec3(.48,.85,.93)*rim,0.,1.);
 // Slightly translucent wing faces echo the original foreground crystals.
 float wing=is_side==0?smoothstep(.10,.23,abs(vuv.x-.5)):0.;
 float alpha=t.a*mix(.97,.84,wing);
 outColor=vec4(rgb,alpha);}
'''

def depth(x,y):
 core=np.clip(1-(x/.24)**2,0,1)*np.clip(1-(y/.45)**2,0,1)
 return .006+.045*np.sqrt(core)

def make_mesh():
 im=Image.open(R/'assets/cocoon_front.png').convert('RGBA');tw,th=im.size;aspect=th/tw
 data=[];nx,ny=100,42
 def v(x,y,side):
  z=side*depth(x,y);e=.0002;dx=(depth(x+e,y)-depth(x-e,y))/(2*e);dy=(depth(x,y+e)-depth(x,y-e))/(2*e)
  n=np.array([-dx,-dy,1.])*side;n/=np.linalg.norm(n)
  u=(x+1)/2;vv=(aspect-y)/(2*aspect)
  return [x,y,z,(u if side==1 else 1-u),vv,*n,1,1,1]
 for side in (1,-1):
  tris=[]
  for j in range(ny):
   y0=aspect-2*aspect*j/ny;y1=aspect-2*aspect*(j+1)/ny
   for i in range(nx):
    x0=-1+2*i/nx;x1=-1+2*(i+1)/nx
    a,b,c,d=v(x0,y0,side),v(x1,y0,side),v(x1,y1,side),v(x0,y1,side)
    tris.extend([a,b,c,a,c,d])
  data.append(np.array(tris,'f4'))
 alpha=np.array(im)[:,:,3];contours,_=cv2.findContours((alpha>100).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 sides=[]
 for contour in contours:
  pts=cv2.approxPolyDP(contour,1.6,True)[:,0,:]
  if len(pts)<3:continue
  for k in range(len(pts)):
   a,b=pts[k],pts[(k+1)%len(pts)];x0,y0=2*a[0]/tw-1,aspect*(1-2*a[1]/th);x1,y1=2*b[0]/tw-1,aspect*(1-2*b[1]/th)
   n=np.array([y1-y0,x0-x1,0]);n/=max(1e-6,np.linalg.norm(n));col=[.22,.26,.32] if abs((x0+x1)/2)<.23 else [.38,.73,.8]
   def vv(x,y,s):return [x,y,s*depth(x,y),0,0,*n,*col]
   A,B,C,D=vv(x0,y0,1),vv(x1,y1,1),vv(x1,y1,-1),vv(x0,y0,-1);sides.extend([A,B,C,A,C,D])
 data.append(np.array(sides,'f4'))
 return data

def main():
 ctx=moderngl.create_standalone_context(backend='egl');prog=ctx.program(vertex_shader=VERT,fragment_shader=FRAG)
 ctx.enable(moderngl.DEPTH_TEST|moderngl.BLEND)
 ctx.blend_func=(moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA,
                 moderngl.ONE,moderngl.ONE_MINUS_SRC_ALPHA)
 tex=[]
 for name in ('front','back'):
  im=Image.open(R/f'assets/cocoon_{name}.png').convert('RGBA')
  # Sample a restrained texture resolution to suppress sharp photographic
  # product microdetail while preserving all identifying design lines.
  im=im.resize((960,round(im.height*960/im.width)),Image.Resampling.LANCZOS)
  t=ctx.texture(im.size,4,im.tobytes());t.filter=(moderngl.LINEAR,moderngl.LINEAR);tex.append(t)
 meshes=make_mesh();vaos=[]
 for a in meshes:
  b=ctx.buffer(a.tobytes());vaos.append(ctx.vertex_array(prog,[(b,'3f 2f 3f 3f','pos','uv','normal','color')]))
 rb=ctx.renderbuffer((W,H),components=4,samples=4);db=ctx.depth_renderbuffer((W,H),samples=4);fbo=ctx.framebuffer(rb,db)
 resolved=ctx.simple_framebuffer((W,H),components=4);out=R/'assets/prop_rotation';out.mkdir(exist_ok=True)
 prog['scale']=470.;prog['center']=(2*1450/W-1,1-2*490/H)
 manifest=[]
 for i in range(70):
  angle=math.radians(-18+205*i/69);rx=math.radians(8);rz=math.radians(-9+3*math.sin(i/69*math.pi))
  X=np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
  Y=np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])
  Z=np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
  mat=Z@Y@X;prog['rotation'].write(mat.T.astype('f4').tobytes());fbo.use();fbo.clear(0,0,0,0,depth=1.)
  prog['is_side']=1;vaos[2].render(moderngl.TRIANGLES)
  prog['is_side']=0
  for v,t in zip(vaos[:2],tex):t.use(0);v.render(moderngl.TRIANGLES)
  ctx.copy_framebuffer(resolved,fbo);im=Image.frombytes('RGBA',(W,H),resolved.read(components=4,alignment=1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
  # OpenGL stores premultiplied RGB; PNG/Pillow require straight-alpha RGB.
  rgba=np.array(im);alpha=rgba[:,:,3:4].astype('float32')/255
  rgba[:,:,:3]=np.clip(rgba[:,:,:3].astype('float32')/np.maximum(alpha,1e-6),0,255).astype('uint8')
  rgba[rgba[:,:,3]==0,:3]=0
  Image.fromarray(rgba,'RGBA').save(out/f'{i:02}.png')
  manifest.append(dict(frame=i,rotation_y_degrees=-18+205*i/69))
 (R/'prop_rotation.json').write_text(json.dumps(dict(renderer=ctx.info['GL_RENDERER'],geometry='front/back height fields and extruded alpha-contour sidewalls',style='cold ambient fill, broad cel shading, restrained metallic detail, translucent wings',poses=manifest),indent=2))
 np.savez_compressed(R/'assets/cocoon_mesh.npz',front=meshes[0],back=meshes[1],side=meshes[2])
 print('Rendered 70 extruded 3D device frames',flush=True)

if __name__=='__main__':main()
