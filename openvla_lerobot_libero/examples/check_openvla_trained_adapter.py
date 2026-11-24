"""Load a saved LoRA in a fresh Python process and check forward + select_action."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.openvla.configuration_openvla import OpenVLAConfig
from lerobot.policies.openvla.training_openvla import TrainableOpenVLAPolicy,NORM_KEY

PROJECT=Path("/home/jovyan/projects/lerobot-openvla")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path)
    args=parser.parse_args()
    run=args.run or Path(json.loads((PROJECT/"outputs/openvla_training/latest_connection_run.json").read_text())["output"])
    config=json.loads((run/"run_config.json").read_text())
    result=json.loads((run/"result.json").read_text())
    base_info=json.loads((run/"adapter/base_model.json").read_text())
    stats=json.loads((run/"adapter/action_normalization.json").read_text())[NORM_KEY]["action"]
    rank=json.loads((run/"adapter/adapter_config.json").read_text())["r"]
    torch.set_num_threads(4)
    policy=TrainableOpenVLAPolicy(OpenVLAConfig(model_id=base_info["checkpoint"],unnorm_key="bridge_orig",device="cuda"),stats,rank=rank)
    base=policy.model.unload()
    policy.model=PeftModel.from_pretrained(base,run/"adapter",is_trainable=False)
    policy.eval()
    data=LeRobotDataset("local/openvla-libero-bowl-validation",root=Path(config["data_root"])/"validation")
    sample=data[config["validation_frame_indices"][0]]
    batch=policy.prepare_training_batch(sample)
    with torch.no_grad(): loss,metrics=policy(batch)
    assert abs(float(loss)-result["adapter_reload_loss_before"])<1e-4
    raw=sample[policy.config.image_key].permute(1,2,0).numpy()
    image=Image.fromarray(np.rint(raw*255).astype(np.uint8)).resize((224,224),Image.Resampling.BICUBIC)
    tensor=torch.from_numpy(np.array(image).copy()).permute(2,0,1).float().div(255).unsqueeze(0)
    action=policy.select_action({policy.config.image_key:tensor,"task":[sample["task"]]})
    assert action.shape==(1,7) and bool(torch.isfinite(action).all())
    report={"fresh_process_adapter_load":True,"loss_matches_saved_run":True,"loss":float(loss),
            "select_action_shape":list(action.shape),"select_action_finite":True,"action":action.cpu().tolist(),
            "simulator_rollout":False,"note":"One resized held-out source image; valid action prediction, not a success-rate measurement"}
    (run/"fresh_process_check.json").write_text(json.dumps(report,indent=2)+"\n")
    print("FRESH_PROCESS_CHECK_COMPLETE",json.dumps(report),flush=True)

if __name__=="__main__": main()
