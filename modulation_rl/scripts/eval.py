import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter

import copy
from pathlib import Path
from modulation.utils import parse_args
from modulation.dotdict import DotDict
from modulation.myray.ray_utils import get_normal_task_config
from modulation.utils import env_creator, episode_is_success


def make_env(env_id, rank):
    def _init():
        env = gym.make(env_id)
        return env
    return _init


main_path = Path(__file__).parent.absolute()
_, _, args, _ = parse_args((main_path), framework='ray')
wandb_config = DotDict(args)
ray_config = {
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "seed": wandb_config.seed,
        # "callbacks": TrainCallback,
        # --------------------------------
        # Training
        # --------------------------------
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        # TODO: new ray version divides also by num_workers -> multiply by num_workers to have same
        "training_intensity": wandb_config.training_intensity * wandb_config.batch_size,
        # Number of env steps to optimize for before returning.
        # This should not be impacting training at all, but only how often we log, checkpoint, etc.
        "timesteps_per_iteration": 1000,
        # --------------------------------
        # Env
        # --------------------------------
        "env": 'modulation_rl_env',
        # ignored atm
        "env_config": {
            # not sure how I could make this unique across workers
            "node_handle": "train_env",
            "env": wandb_config.env,
            "task": wandb_config.task,
            "penalty_scaling": wandb_config.penalty_scaling,  # tune.choice([0.0, 0.1])
            "acceleration_penalty": wandb_config.acceleration_penalty,
            "time_step": wandb_config.time_step,
            "seed": wandb_config.seed,
            "world_type": wandb_config.world_type,
            "init_controllers": wandb_config.init_controllers,
            "learn_vel_norm": wandb_config.learn_vel_norm,  # tune.choice([0.1, 0.3, 0.5])
            "collision_penalty": wandb_config.collision_penalty,  # tune.choice([5, 10, 25])
            "vis_env": wandb_config.vis_env,
            "transition_noise_base": wandb_config.transition_noise_base,  # tune.choice([0.005, 0.01, 0.015]),
            "ikslack_dist": wandb_config.ikslack_dist,
            "ikslack_rot_dist": wandb_config.ikslack_rot_dist,
            "ikslack_sol_dist_reward": wandb_config.ikslack_sol_dist_reward,
            "ikslack_penalty_multiplier": wandb_config.ikslack_penalty_multiplier,
            "ik_fail_thresh": wandb_config.ik_fail_thresh,
            "use_map_obs": wandb_config.use_map_obs,
            "global_map_resolution": wandb_config.global_map_resolution,
            "local_map_resolution": wandb_config.local_map_resolution,
            "overlay_plan": wandb_config.overlay_plan,  # tune.choice([True, False])
            "concat_plan": wandb_config.concat_plan,
            "concat_prev_action": wandb_config.concat_prev_action,
            "gamma": wandb_config.gamma,
            "obstacle_config": wandb_config.obstacle_config,
            "simpleobstacle_spacing": wandb_config.simpleobstacle_spacing,  # tune.choice([1.5, 1.75])
            "simpleobstacle_offsetstd": wandb_config.simpleobstacle_offsetstd,  # tune.choice([1.5, 1.75])
            "eval": False,
            "frame_skip": wandb_config.frame_skip,
            "frame_skip_observe": wandb_config.frame_skip_observe,
            "frame_skip_curriculum": wandb_config.frame_skip_curriculum,
            "start_level": wandb_config.frame_skip,
            "algo": wandb_config.algo,
            "use_fwd_orientation": wandb_config.use_fwd_orientation,
            "iksolver": wandb_config.iksolver,
            "selfcollision_as_failure": wandb_config.selfcollision_as_failure,
            "bioik_center_joints_weight": wandb_config.bioik_center_joints_weight,
            "bioik_avoid_joint_limits_weight": wandb_config.bioik_avoid_joint_limits_weight,
            "bioik_regularization_weight": wandb_config.bioik_regularization_weight,
            "bioik_regularization_type": wandb_config.bioik_regularization_type,
            "learn_torso": wandb_config.learn_torso,
            "learn_joint_values": wandb_config.learn_joint_values,
        },
        # "env_task_fn": frame_skip_curriculum_fn if wandb_config.frame_skip_curriculum else None,
        # If True, RLlib will learn entirely inside a normalized action space
        # (0.0 centered with small stddev; only affecting Box components) and
        # only unsquash actions (and clip just in case) to the bounds of
        # env's action space before sending actions back to the env.
        "normalize_actions": True,
        # Whether to clip rewards during Policy's postprocessing.
        # None (default): Clip for Atari only (r=sign(r)).
        # True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.
        # False: Never clip.
        # [float value]: Clip at -value and + value.
        # Tuple[value1, value2]: Clip at value1 and value2.
        "clip_rewards": None,
        # If True, RLlib will clip actions according to the env's bounds
        # before sending them back to the env.
        # NOTE: (sven) This option should be obsoleted and always be False.
        "clip_actions": False,
        # Which observation filter to apply to the observation.
        "observation_filter": 'NoFilter',  # tune.choice(["NoFilter", "MeanStdFilter"])
        # Disable setting done=True at end of episode. This should be set to True
        # for infinite-horizon MDPs (e.g., many continuous control problems).
        "no_done_at_end": wandb_config.no_done_at_end
    }

task_config_updates = get_normal_task_config(wandb_config=wandb_config)
ray_config.update(task_config_updates)

rob_env = env_creator(ray_config["env_config"])

torch.manual_seed(ray_config["seed"])
np.random.seed(ray_config["seed"])

# Parameters for training
target_update_interval = 1
num_steps = 5000001
eval = True
eval_num_reach_goal = []
eval_num_perfect_succ = []
eval_nr_base_collisions = []
eval_nr_kin_failures = []
print("#########OK6############")
# Agent
agent = SAC(rob_env.observation_space, rob_env.action_space, target_update_interval, ray_config, num_steps)
print("#########OK7############")
agent.load_checkpoint("/home/ros/kjx/LLB/new_n2m2/src/modulation_rl/scripts/checkpoints_pnp/sac_checkpoint_husky_ur5_10000", evaluate=True)
print("#########OK8############")
#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), ray_config["env"],
                                                             "Gaussian", "autotune" if False else ""))

if eval is True:
    avg_reward = 0.
    episodes = 100
    for _  in range(episodes):
        state = rob_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, eval = True)

            next_state, reward, done, info = rob_env.step(action)
            episode_reward += reward

            if done:
                if info["ee_done"]:
                    eval_num_reach_goal.append(1)
                else:
                    eval_num_reach_goal.append(0)
                if episode_is_success(info["nr_kin_failures"], info["nr_base_collisions"], info["ee_done"]):
                    eval_num_perfect_succ.append(1)
                else:
                    eval_num_perfect_succ.append(0)
                eval_nr_base_collisions.append(info["nr_base_collisions"])
                eval_nr_kin_failures.append(info["nr_kin_failures"])
                    
            state = next_state
        avg_reward += episode_reward
    avg_reward /= episodes
    
    # writer.add_scalar('eval/avg_reward_test', avg_reward, i_episode)

    if len(eval_num_reach_goal) >=100:
        # writer.add_scalar('eval/base_collisions', np.mean(eval_nr_base_collisions), i_episode)
        # writer.add_scalar('eval/kin_failures', np.mean(eval_nr_kin_failures), i_episode)
        # writer.add_scalar('eval/goal_reached', np.mean(eval_num_reach_goal), i_episode)
        # writer.add_scalar('eval/perfect_success', np.mean(eval_num_perfect_succ), i_episode)

        print("eval_nr_base_collisions:")
        print(np.mean(eval_nr_base_collisions))
        print("eval_nr_kin_failures:")
        print(np.mean(eval_nr_kin_failures))
        print("eval_num_reach_goal:")
        print(np.mean(eval_num_reach_goal))
        print("eval_num_perfect_succ:")
        print(np.mean(eval_num_perfect_succ))

        eval_nr_base_collisions = []
        eval_nr_kin_failures = []
        eval_num_reach_goal = []
        eval_num_perfect_succ = []

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
    print("----------------------------------------")

rob_env.close()

