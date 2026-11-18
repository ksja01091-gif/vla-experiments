from pathlib import Path
import mujoco
import mujoco.viewer

root = Path.home() / "mujoco_menagerie"
scene = mujoco.MjSpec.from_file(str(root / "universal_robots_ur5e/scene.xml"))
gripper = mujoco.MjSpec.from_file(str(root / "robotiq_2f85/2f85.xml"))

scene.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
scene.option.impratio = 10
scene.attach(gripper, site=scene.site("attachment_site"), prefix="gripper/")

block = scene.worldbody.add_body(name="test_block", pos=[0.45, 0, 0.2])
block.add_freejoint(name="test_block_joint")
block.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[0.025, 0.025, 0.025],
    mass=0.05,
    rgba=[1, 0.4, 0.1, 1],
)

model = scene.compile()
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_block_joint")
address = model.jnt_qposadr[joint_id]
data.qpos[address:address + 3] = [0.45, 0, 0.2]
mujoco.mj_forward(model, data)
mujoco.viewer.launch(model, data)

