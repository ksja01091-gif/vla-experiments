# VLA experiments

Code and small examples from my robot learning experiments. The first example uses the public LIBERO Spatial bowl task with OpenVLA-7B.

## OpenVLA: one LIBERO scene

See [`openvla_one_scene/OpenVLA_base_LIBERO_one_scene.ipynb`](openvla_one_scene/OpenVLA_base_LIBERO_one_scene.ipynb). It reads one scene image and a task instruction, predicts seven normalized action values, and converts them to a candidate LIBERO controller action using action statistics from 20 training demonstrations. The source episode comes from the public `openvla/modified_libero_rlds` dataset; the two sample images are included for inspection.

This example checks one action prediction. It does **not** run a LIBERO simulation or measure task success. It does not fine-tune OpenVLA.

- `base_openvla_one_scene.py`: load OpenVLA and predict seven action values.
- `convert_openvla_action.py`: convert predicted values using the training statistics.
- `action_stats.json`: action statistics from the earlier train20 experimental split.
- `inspect_libero_tfrecord.py`: inspect a locally downloaded LIBERO TFRecord episode.
- `make_openvla_base_notebook.py`: generate the Colab notebook.
- `libero_sample/`: two example scene images.

The full model weights, complete public dataset, checkpoints, and GPU outputs are kept outside Git. The notebook was run in Colab on an A100; rerunning it downloads the model and installs its required packages.

## UR5e scripted demonstration collection

See [`ur5e_scripted_demos/README.md`](ur5e_scripted_demos/README.md) for the code and result of the 40-position MuJoCo block-grasp demonstration run. This is scripted data generation; the data has not yet been converted for OpenVLA training.
