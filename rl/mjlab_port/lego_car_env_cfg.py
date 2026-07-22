"""mjlab environment configuration for the LEGO signature car.

Ports lego_car.xml (see that file's header for how it differs from the
original lego_car_with_pencil.xml) into mjlab's manager-based env API, so a
fleet of cars can be simulated in one batched MuJoCo-Warp sim and viewed in
the browser viser viewer (see play_car.py).

Scope: this is the *visualization / scaffolding* port. The observations,
reward, and episode logic are a minimal drive-forward task, NOT the
signature-tracing MDP from rl/signature_env.py - porting that (path
following, accuracy-gated progress, per-env target paths) is a separate
step on top of this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  base_ang_vel,
  base_lin_vel,
  joint_vel_rel,
  last_action,
  reset_joints_by_offset,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_LEGO_CAR_XML: Path = Path(__file__).parent / "lego_car.xml"

_CAR_CFG = SceneEntityCfg("lego_car")
_WHEELS_CFG = SceneEntityCfg("lego_car", joint_names=("joint_left", "joint_right"))


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_LEGO_CAR_XML))


def _get_lego_car_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=("joint_left", "joint_right")),),
    ),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.02),
      joint_pos={".*": 0.0},
      joint_vel={".*": 0.0},
    ),
  )


# Rewards.


def forward_speed(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _CAR_CFG,
) -> torch.Tensor:
  """Chassis-frame forward (x) velocity, m/s. Placeholder task reward that
  makes zero/random/trained agents all produce sensible learning signal."""
  asset: Entity = env.scene[asset_cfg.name]
  return base_lin_vel(env, asset_cfg)[:, 0]


# Environment config.


def lego_car_env_cfg(play: bool = False, num_envs: int = 16) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=base_lin_vel, params={"asset_cfg": _CAR_CFG}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=base_ang_vel, params={"asset_cfg": _CAR_CFG}
    ),
    "wheel_vel": ObservationTermCfg(
      func=joint_vel_rel, params={"asset_cfg": _WHEELS_CFG}
    ),
    "last_action": ObservationTermCfg(func=last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "wheel_effort": JointEffortActionCfg(
      entity_name="lego_car",
      actuator_names=("joint_left", "joint_right"),
      scale=1.0,
    ),
  }

  events = {
    # This event also applies the per-env grid offset (env_origins), which is
    # what spreads the fleet out in the shared viewer scene.
    "reset_root": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.05, 0.05),
          "y": (-0.05, 0.05),
          "yaw": (-3.1416, 3.1416),
        },
        "velocity_range": {},
        "asset_cfg": _CAR_CFG,
      },
    ),
    "reset_wheels": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": _WHEELS_CFG,
      },
    ),
  }

  rewards = {
    "forward_speed": RewardTermCfg(
      func=forward_speed,
      weight=1.0,
      params={"asset_cfg": _CAR_CFG},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"lego_car": _get_lego_car_cfg()},
      num_envs=num_envs,
      env_spacing=0.35,  # paper-sized cells: 199x137mm sheet + margin
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="lego_car",
      body_name="chassis",
      distance=0.6,
      elevation=-25.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.002),  # matches the original model
    ),
    decimation=10,  # 50 Hz control, same as rl/signature_env.py's frame skip
    episode_length_s=30.0,
  )
  if play:
    cfg.episode_length_s = 1e10
  return cfg


# RL config (used by mjlab's train path; the play viewer only needs the env).


def lego_car_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="lego_car",
    logger="tensorboard",  # no W&B login required
    save_interval=50,
    num_steps_per_env=32,
    max_iterations=500,
  )
