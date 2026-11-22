
from pathlib import Path
import json, numpy as np, torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import load_pretrained_policy_processors

run = Path('/content/drive/MyDrive/VLA_experiments/UR5e_Pi05_40demo_1epoch_seed7')
checkpoint = run/'pretrained_model'
data = np.load(run/'validation_scene_0_input.npz', allow_pickle=False)
observation = {
    'observation.images.image': torch.from_numpy(data['image']).float(),
    'observation.state': torch.from_numpy(data['state']).float(),
    'task': str(data['task'].item()),
}
assert observation['observation.images.image'].shape == (3, 256, 256)
assert observation['observation.state'].shape == (7,)
assert torch.cuda.is_available()
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
preprocess, postprocess = load_pretrained_policy_processors(checkpoint)
policy = PI05Policy.from_pretrained(checkpoint, local_files_only=True)
policy.to(device='cuda')
policy.eval()
with torch.inference_mode():
    model_input = preprocess(observation)
    raw_action = policy.select_action(model_input)
    action = postprocess(raw_action).detach().cpu().float()
assert action.numel() == 7, (tuple(raw_action.shape), tuple(action.shape))
assert bool(torch.isfinite(action).all()), action
result = {
    'checkpoint': str(checkpoint),
    'input_file': str(run/'validation_scene_0_input.npz'),
    'task': observation['task'],
    'image_shape': list(observation['observation.images.image'].shape),
    'state_shape': list(observation['observation.state'].shape),
    'raw_action_shape': list(raw_action.shape),
    'output_shape': list(action.shape),
    'all_finite': True,
    'action': [round(float(x), 6) for x in action.reshape(-1)],
}
(run/'fresh_action_result.json').write_text(json.dumps(result, indent=2))
print('FRESH_PROCESS_ACTION_OK', json.dumps(result), flush=True)
