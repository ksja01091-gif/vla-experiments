"""Read one public demonstration frame through the REAL preprocessing, without GPU weights."""
import json,types
from comparison_common import *
from transformers import AutoProcessor
from lerobot.policies.openvla.training_openvla import TrainableOpenVLAPolicy,load_reference
train,val,stats=exp.get_data();sample=train[0]
config=OpenVLAConfig(model_id="openvla/openvla-7b",unnorm_key="bridge_orig",device="cuda:0")
snapshot=json.loads((ROOT/"outputs/ten_epoch_bowl/20261104T075816_073454Z/training_config.json").read_text())["base_snapshot"]
processor=AutoProcessor.from_pretrained(snapshot,trust_remote_code=True,local_files_only=True)
action_module,_=load_reference("prismatic/vla/action_tokenizer.py","flow_action_tokens")
prompt_module,_=load_reference("prismatic/models/backbones/llm/prompting/base_prompter.py","flow_prompt")
statistics=dict(stats);statistics["mask"]=[True]*6+[False]
owner=types.SimpleNamespace(config=config,processor=processor,action_statistics=statistics,
action_tokenizer=action_module.ActionTokenizer(processor.tokenizer),prompt_builder_class=prompt_module.PurePromptBuilder)
owner.normalize_action=types.MethodType(TrainableOpenVLAPolicy.normalize_action,owner)
batch=TrainableOpenVLAPolicy.prepare_training_batch(owner,sample)
pixels=np.rint(sample[config.image_key].permute(1,2,0).numpy()*255).astype(np.uint8)
Image.fromarray(pixels).save(RUN/"one_demo_frame.png")
record={"data":"public LIBERO selected/preprocessed experimental split","train_index":0,
"task":sample["task"],"raw_image_shape":list(sample[config.image_key].shape),
"demonstration_action_not_prediction":sample["action"].tolist(),
"normalized_demonstration_action":owner.normalize_action(sample["action"].numpy()).tolist(),
"gold_action_token_ids":batch["labels"][0,-8:-1].tolist(),
"EOS_token_id":int(batch["labels"][0,-1]),"ignored_prompt_label":-100,
"prepared_shapes":{k:list(v.shape) for k,v in batch.items()},"model_weights_loaded":False,
"purpose":"trace real data preprocessing; no model prediction or optimization"}
write(RUN/"one_sample_flow.json",record)
print("REAL_PREPROCESSING_WALKTHROUGH",json.dumps(record),flush=True)
