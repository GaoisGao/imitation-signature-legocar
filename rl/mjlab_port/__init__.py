"""mjlab port of the LEGO signature car. Importing this package registers
the tasks with mjlab's registry (mirrors how mjlab's own tasks register).

Tasks:
  Mjlab-LegoCar-Drive     - minimal drive-forward task (visualization scaffolding)
  Mjlab-LegoCar-Signature - the signature-tracing MDP (see signature_mdp.py)
"""

from lego_car_env_cfg import (
  lego_car_env_cfg,
  lego_car_ppo_runner_cfg,
)
from signature_env_cfg import (
  lego_car_signature_env_cfg,
  lego_car_signature_ppo_runner_cfg,
)

from mjlab.tasks.registry import register_mjlab_task

TASK_ID = "Mjlab-LegoCar-Drive"
SIGNATURE_TASK_ID = "Mjlab-LegoCar-Signature"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=lego_car_env_cfg(),
  play_env_cfg=lego_car_env_cfg(play=True),
  rl_cfg=lego_car_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id=SIGNATURE_TASK_ID,
  env_cfg=lego_car_signature_env_cfg(),
  play_env_cfg=lego_car_signature_env_cfg(play=True),
  rl_cfg=lego_car_signature_ppo_runner_cfg(),
)
