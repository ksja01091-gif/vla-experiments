"""Reuse preserved train-only importance and physically export 90/80/70 on CPU."""
from pathlib import Path
import json, hashlib
from pruned_policy import RUN, CHECKPOINT, torch, load_file, save_file, write

config=json.loads((RUN/"experiment_config.json").read_text())
old=Path(config["parent_pruning_run"])
scores=json.loads((RUN/"calibration_importance.json").read_text())["importance"]
modules=json.loads((old/"checkpoints/sp_pruned96/adapter_config.json").read_text())["modules"]
assert set(modules)==set(scores) and len(modules)==64
original=load_file(str(CHECKPOINT/"adapter_model.safetensors"),device="cpu")
source_hash=hashlib.sha256((CHECKPOINT/"adapter_model.safetensors").read_bytes()).hexdigest()
exports=[]
for rank in [90,80,70]:
    dest=RUN/"checkpoints"/f"sp_pruned{rank}"
    assert not dest.exists(),"Existing checkpoint preserved"
    dest.mkdir(parents=True)
    weights={};plan={}
    for name in modules:
        score=torch.tensor(scores[name],dtype=torch.float64)
        assert score.shape==(128,) and torch.isfinite(score).all()
        keep=torch.argsort(score,descending=True,stable=True)[:rank].sort().values
        plan[name]={"keep_indices":keep.tolist(),"retained_calibration_importance":float(score[keep].sum()/score.sum()),
                    "original_rank":128,"physical_rank":rank}
        for key,value in original.items():
            if not key.startswith(name+"."):continue
            if key.endswith(".sp_A.weight"):value=value.index_select(0,keep)
            elif key.endswith(".sp_B.weight"):value=value.index_select(1,keep)
            weights[key]=value.contiguous()
        assert weights[name+".sp_A.weight"].shape[0]==rank
        assert weights[name+".sp_B.weight"].shape[1]==rank
    assert set(weights)==set(original)
    cfg={"method":"static-physical-SP-bank-pruning","original_rank":128,"physical_rank":rank,"eta":.9,
         "router_full_128_retained":True,"selection":"original 128-score eta mask, then retained component indices",
         "static_pruning_changes_policy":True,"guaranteed_90pct_energy_after_removal":False,
         "importance_method":"train-only squared component output norm heuristic","modules":plan,
         "source_checkpoint":str(CHECKPOINT),"source_adapter_file_sha256":source_hash,
         "additional_training":False,"trained_epochs":5,"calibration_reused_from":str(old)}
    write(dest/"adapter_config.json",cfg)
    save_file(weights,str(dest/"adapter_model.safetensors"))
    for file in ["action_normalization.json","base_model.json"]:(dest/file).write_bytes((CHECKPOINT/file).read_bytes())
    h=hashlib.sha256()
    h.update(json.dumps({n:plan[n]["keep_indices"] for n in plan},sort_keys=True).encode())
    for name,t in sorted(weights.items()):h.update(name.encode());h.update(t.float().numpy().tobytes())
    (dest/"adapter_sha256.txt").write_text(h.hexdigest())
    exports.append({"variant":f"sp_pruned{rank}","physical_rank":rank,"checkpoint":str(dest),
                    "adapter_file_bytes":(dest/"adapter_model.safetensors").stat().st_size,
                    "adapter_parameters":sum(t.numel() for t in weights.values())})
write(RUN/"export_complete.json",{"complete":True,"calibration_reused":True,"validation_used":False,"exports":exports})
print("EXPORTED_THREE_RANKS",json.dumps(exports),flush=True)
