"""Select whole target-task demonstrations from the pinned OpenVLA RLDS source.
Run with the isolated openvla_data310 interpreter. No model training occurs.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"
SEED = 7


def normalize(text):
    return " ".join(text.lower().strip().rstrip(".").split())


def official_action_transform():
    root = Path("/home/jovyan/projects/openvla/prismatic/vla/datasets/rlds")
    paths = {root/"utils/data_utils.py": ["invert_gripper_actions"],
             root/"oxe/transforms.py": ["libero_dataset_transform"]}
    namespace = {"tf":tf,"Dict":Dict,"Any":Any}
    provenance = {}
    for path,names in paths.items():
        source = path.read_text()
        nodes = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes}!=set(names):
            raise RuntimeError("Official action transform was not found.")
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
        provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return namespace["libero_dataset_transform"],provenance


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    archive_dir=output/"candidate_episodes"
    archive_dir.mkdir(parents=True,exist_ok=False)
    transform,provenance=official_action_transform()
    builder=tfds.builder_from_directory(str(args.raw_dir))
    expected=builder.info.splits["train"].num_examples
    options=tf.data.Options()
    options.experimental_deterministic=True
    options.threading.private_threadpool_size=2
    dataset=builder.as_dataset(split="train",shuffle_files=False).with_options(options)
    candidates=[];duplicates=[];ids=set();task_counts={};seen=0
    for ordinal,episode in enumerate(dataset):
        seen+=1
        iterator=iter(tfds.as_numpy(episode["steps"].with_options(options)))
        first=next(iterator)
        instruction=first["language_instruction"].decode("utf-8")
        task_counts[instruction]=task_counts.get(instruction,0)+1
        if normalize(instruction)!=normalize(TASK):
            if seen%50==0:
                print("SCAN:",seen,"of",expected,"target demos:",len(candidates),flush=True)
            continue
        steps=[first,*iterator]
        if not all(normalize(step["language_instruction"].decode("utf-8"))==normalize(TASK) for step in steps):
            raise RuntimeError("Mixed task labels inside one episode.")
        images=np.stack([s["observation"]["image"] for s in steps]).astype(np.uint8)
        raw=np.stack([s["action"] for s in steps]).astype(np.float32)
        states=np.stack([s["observation"]["state"] for s in steps]).astype(np.float32)
        joints=np.stack([s["observation"]["joint_state"] for s in steps]).astype(np.float32)
        count=len(steps)
        if images.shape!=(count,256,256,3) or raw.shape!=(count,7) or states.shape!=(count,8) or joints.shape!=(count,7):
            raise RuntimeError("Unexpected source shapes.")
        if count<2 or not all(np.isfinite(x).all() for x in [raw,states,joints]):
            raise RuntimeError("Invalid demonstration values.")
        standardized=transform({"action":tf.constant(raw),"observation":{"state":tf.constant(states)}})["action"].numpy()
        if not np.array_equal(raw[:,:6],standardized[:,:6]):
            raise RuntimeError("Arm actions changed unexpectedly.")
        if not np.array_equal(standardized[:,-1],1-np.clip(raw[:,-1],0,1)):
            raise RuntimeError("Gripper does not match the reference transform.")
        source_path=episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
        digest=hashlib.sha256()
        digest.update(source_path.encode());digest.update(images.tobytes());digest.update(raw.tobytes());digest.update(states.tobytes());digest.update(joints.tobytes())
        identity=digest.hexdigest()
        if identity in ids:
            duplicates.append({"ordinal":ordinal,"content_id":identity});continue
        ids.add(identity)
        archive=archive_dir/(identity+".npz")
        with archive.open("wb") as handle:
            np.savez_compressed(handle,images=images,actions_raw=raw,actions=standardized,
                                states=states,joints=joints,
                                rewards=np.asarray([s["reward"] for s in steps],dtype=np.float32),
                                is_first=np.asarray([s["is_first"] for s in steps],dtype=bool),
                                is_last=np.asarray([s["is_last"] for s in steps],dtype=bool),
                                is_terminal=np.asarray([s["is_terminal"] for s in steps],dtype=bool))
        record={"content_id":identity,"source_ordinal":ordinal,"source_provenance_path":source_path,
                "instruction":instruction,"frames":count,"archive":str(archive),
                "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
                "image_shape":[256,256,3],"raw_gripper_values":np.unique(raw[:,-1]).tolist(),
                "standardized_gripper_values":np.unique(standardized[:,-1]).tolist()}
        candidates.append(record)
        if len(candidates)%5==0:
            print("TARGET_DEMOS_FOUND:",len(candidates),"scanned",seen,"of",expected,flush=True)
    if seen!=expected:
        raise RuntimeError(f"Expected {expected} episodes, read {seen}.")
    inventory={"total_source_episodes":seen,"target_task":TASK,"target_unique_episodes":len(candidates),
               "task_counts":task_counts,"duplicates":duplicates,"candidates":candidates,
               "reference_transform_sha256":provenance,"tfds_version":tfds.__version__,"tensorflow_version":tf.__version__}
    (output/"inventory.json").write_text(json.dumps(inventory,indent=2,ensure_ascii=False)+"\n")
    if len(candidates)<25:
        raise RuntimeError(f"Only {len(candidates)} usable target episodes; 25 requested.")
    ordered=sorted(candidates,key=lambda row:row["content_id"])
    permutation=np.random.default_rng(SEED).permutation(len(ordered))
    selected=[ordered[int(index)] for index in permutation[:25]]
    train,validation=selected[:20],selected[20:25]
    if set(r["content_id"] for r in train)&set(r["content_id"] for r in validation):
        raise RuntimeError("Train/validation episode overlap.")
    all_train=np.concatenate([np.load(r["archive"],allow_pickle=False)["actions"] for r in train])
    stats={"source":"training episodes only","episodes":20,"frames":len(all_train),
           "mean":all_train.mean(axis=0).tolist(),"std":all_train.std(axis=0).tolist(),
           "min":all_train.min(axis=0).tolist(),"max":all_train.max(axis=0).tolist(),
           "q01":np.quantile(all_train,0.01,axis=0).tolist(),"q99":np.quantile(all_train,0.99,axis=0).tolist()}
    (output/"action_statistics_train_only.json").write_text(json.dumps(stats,indent=2)+"\n")
    split={"task":TASK,"seed":SEED,"selection":"sort whole-episode content IDs, then numpy default_rng(7) permutation; no selection using evaluation success",
           "train":train,"validation":validation,"train_episodes":20,"validation_episodes":5,
           "train_frames":sum(r["frames"] for r in train),"validation_frames":sum(r["frames"] for r in validation),
           "episode_overlap":False,"evaluation_initial_state_disjointness":"not established; this is an episode-level data split",
           "image_processing":"source pixels unchanged; no extra rotation/crop/resize during export",
           "action_processing":"official libero_dataset_transform; first six unchanged, gripper=1-clip(raw,0,1)",
           "extra_source_fields":"wrist images remain in raw RLDS but are not exported; current OpenVLA baseline uses the external camera only"}
    (output/"split_manifest.json").write_text(json.dumps(split,indent=2,ensure_ascii=False)+"\n")
    print("EXTRACTION_COMPLETE:",len(candidates),"unique target demos; selected 20 train + 5 validation",flush=True)
    print("SELECTED_FRAMES:",split["train_frames"],split["validation_frames"],flush=True)


if __name__=="__main__":
    main()
