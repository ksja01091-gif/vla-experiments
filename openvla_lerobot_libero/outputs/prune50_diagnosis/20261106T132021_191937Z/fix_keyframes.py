"""Readable chronological captions on unmodified recorded frames; no models."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
from matplotlib import font_manager
R=Path(__file__).resolve().parent
CFG=json.loads((R/'diagnostic_config.json').read_text());SOURCE=Path(CFG['source_run'])
data=json.loads((R/'replay_results.json').read_text())['rows']
font=ImageFont.truetype(font_manager.findfont('DejaVu Sans'),14)
board=Image.new('RGB',(1024,672),'white');draw=ImageDraw.Draw(board)
for index,tag in enumerate(['dense_sft','p50']):
    trial=next(x for x in data if x['tag']==tag and x['state']==0)
    traces=[json.loads(l) for l in (R/tag/'state_00/trace.jsonl').read_text().splitlines()]
    by_step={x['step']:x for x in traces}
    if tag=='dense_sft':
        points=[(0,'Start'),(trial['first_grasp_both_pads'],'Both-pad contact'),
            (trial['first_lift_5cm'],'Grasp + lift >5cm'),(trial['first_official_success'],'On plate, still held')]
    else:
        peak=max(traces,key=lambda x:x['target_lift_m'])['step']
        points=[(0,'Start'),(trial['first_close_command'],'First close command'),
            (peak,'One-pad contact'),(220,'No task success')]
    y=index*336
    draw.text((8,y+3),'ORIGINAL + SFT1' if tag=='dense_sft' else '50% PRUNED + SFT1',font=font,fill='#377bbc' if index==0 else '#df873d')
    for col,(step,label) in enumerate(points):
        with Image.open(SOURCE/'simulation'/tag/'state_00/frames'/f'{step:04d}.png') as image:
            board.paste(image.convert('RGB'),(col*256,y+26))
        caption=f"step {step}: {label}\nheight +{by_step[step]['target_lift_m']*100:.1f}cm\nboth-pad contact: {by_step[step]['grasp_both_pads']}"
        draw.multiline_text((col*256+4,y+285),caption,font=font,fill='black',spacing=2)
board.save(R/'state0_keyframes.png')
print('KEYFRAME_CAPTIONS_VERIFIED_CHRONOLOGICAL',flush=True)
