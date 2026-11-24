from pathlib import Path
import json,sys
sys.path.insert(0,'/home/jovyan/projects/lerobot-openvla/examples')
import numpy as np
from PIL import Image
from openvla_eval_preprocessing import prepare_image,REFERENCE_NAMESPACE
r=Path(__file__).parent
for item in json.loads((r/'photo_records.json').read_text()):
 p=r/'photos'/str(item['validation_index'])
 image=np.array(Image.open(p/'source.png').convert('RGB'))
 raw=np.ascontiguousarray(image[::-1,::-1])
 Image.fromarray(prepare_image(raw)).save(p/'original_eval.png')
 Image.fromarray(REFERENCE_NAMESPACE['get_libero_image']({'agentview_image':raw},224)).save(p/'no_crop_eval.png')
print('PHOTO_VARIANTS_COMPLETE')
