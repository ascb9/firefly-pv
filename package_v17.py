#!/usr/bin/env python3
"""Package useful portable project content and preserve the existing cels."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--comparison',required=True,type=Path);args=ap.parse_args()
    dest=HERE/'qa/原PV并排对照.mp4';shutil.copy2(args.comparison,dest)
    refs=HERE/'references/original_landmarks';refs.mkdir(parents=True,exist_ok=True)
    for n in [1643,1660,1667,1691]:
        shutil.copy2(HERE/'qa/source_frames_v17'/f'f{n:05}.png',refs/f'f{n:05}.png')
    excluded={'deps','__pycache__','source_frames_v17','flows'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(HERE.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(HERE)
            if any(x in excluded for x in rel.parts):continue
            if p.suffix=='.pyc':continue
            if rel.parts[0]=='qa' and p.suffix=='.png':continue
            if rel.parts[0]=='qa' and p.suffix=='.mp4' and p!=dest:continue
            z.write(p,Path('firefly_pv_v17')/rel)
    with zipfile.ZipFile(args.output) as z:
        error=z.testzip()
        if error:raise RuntimeError('Archive CRC failed: '+error)
        assets=[n for n in z.namelist() if n.startswith('firefly_pv_v17/assets/') and n.endswith('.png')]
        print(json.dumps(dict(output=str(args.output),bytes=args.output.stat().st_size,
            files=len(z.namelist()),png_assets=len(assets),crc_valid=True),ensure_ascii=False))

if __name__=='__main__':main()
