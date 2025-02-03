import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import argparse
import gymnasium as gym
import numpy as np
import torch
import pybullet as p
import matplotlib.pyplot as plt

from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

class CustomCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.reward_hist = np.empty((0,))
        self.drone_pos_hist = np.empty((0, 3))

    def _on_step(self):
        position = np.array(self.locals['infos'][0]['Drone Position']).flatten()
        if self.drone_pos_hist.size == 0:
            self.drone_pos_hist = position
        else:
            self.drone_pos_hist = np.vstack((self.drone_pos_hist, position))
        return super()._on_step()

    def _on_rollout_end(self):
        # self.model.randomizeTargets()

        # print(self.locals['infos'])

        self.reward_hist = np.concatenate((self.reward_hist, self.locals['rewards']), axis = 0)

        # if(self.num_timesteps % self.save_interval == 0):
        np.savetxt("Reward Multi.csv", self.reward_hist, delimiter = ',')
        np.savetxt("Drone Position Multi.csv", self.drone_pos_hist, delimiter = ',')

        

class RewardLoggerCallback:
    def __init__(self):
        self.episode_rewards = []
        self.x_data = []  # Store episode numbers
        self.y_data = []  # Store rewards
        # Set up the plot
        # plt.ion()  # Enable interactive mode
        # self.fig, self.ax = plt.subplots()
        # self.ax.set_xlabel('Episode')
        # self.ax.set_ylabel('Reward')
        # self.ax.set_title('Real-time Reward Plot')
        # self.ax.legend(['Episode Reward'])
        
    def __call__(self, locals_, globals_):
        # Get reward of the current episode
        print(locals_)
        # episode_reward = locals_['rewards']
        # self.episode_rewards.append(episode_reward)
        
        # # Append the current episode number and reward
        # episode_number = len(self.episode_rewards)
        # self.x_data.append(episode_number)
        # self.y_data.append(episode_reward)

        # Update the plot
        # self.ax.clear()  # Clear the previous plot
        # self.ax.plot(self.x_data, self.y_data, label='Episode Reward')
        # self.ax.set_xlabel('Episode')
        # self.ax.set_ylabel('Reward')
        # self.ax.set_title('Real-time Reward Plot')
        # self.ax.legend()

        # Redraw the plot to show the updated data
        # plt.draw()
        # plt.pause(0.01)  # Pause briefly to allow the plot to update

    def close_plot(self):
        # Finalize the plot when training is done
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep the plot displayed

class MultiAgentNeighborEnv(BaseRLAviary):
    """
        Multi-agent environment for neighbor-based cooperative control of multiple drones.

    """
    
    def __init__(self, drone_model = DroneModel.CF2X, num_drones = 1, neighbourhood_radius = np.inf, initial_xyzs=None, initial_rpys=None, physics = Physics.PYB, pyb_freq = 240, ctrl_freq = 240, gui=False, record=False, obs = ObservationType.KIN, act = ActionType.RPM, neighbor_selection = 'closest'):
        super().__init__(drone_model, num_drones, neighbourhood_radius, initial_xyzs, initial_rpys, physics, pyb_freq, ctrl_freq, gui, record, obs, act)

        self.neighbor_selection = neighbor_selection

        self.last_distances = np.array([0 for _ in range(num_drones)])

        self._cohesionReward = .1

        self._droneCollisionReward = -5

        self._targetReachedReward = 2

        self._safeDistance = .5

        self.EPISODE_LEN_SEC = 15

        self.target_xyz = np.array([0, 0, 0])

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12 + TARGET_DIMENSIONS) depending on the observation type.
        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                            high=255,
                            shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo, lo, 0, lo, lo, lo, lo, lo, lo, lo, lo, lo] for _ in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi] for _ in range(self.NUM_DRONES)])

            # Add action buffer to observation space
            act_lo = -1
            act_hi = +1
            for _ in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo, act_lo, act_lo, act_lo] for _ in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi, act_hi, act_hi, act_hi] for _ in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE == ActionType.PID:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo, act_lo, act_lo] for _ in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi, act_hi, act_hi] for _ in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for _ in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for _ in range(self.NUM_DRONES)])])

            # Add single target position to observation space
            target_lower_bound = np.array([lo, lo, lo])  # Target position (X, Y, Z) bounds
            target_upper_bound = np.array([hi, hi, hi])

            # Reshape target bounds to match the number of drones
            target_lower_bound = np.tile(target_lower_bound, (self.NUM_DRONES, 1))
            target_upper_bound = np.tile(target_upper_bound, (self.NUM_DRONES, 1))

            # Concatenate target bounds
            obs_lower_bound = np.hstack([obs_lower_bound, target_lower_bound])
            obs_upper_bound = np.hstack([obs_upper_bound, target_upper_bound])

            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")

    def cohesionReward(self):
        positions = self.pos
        centroid = np.mean(positions, axis = 0)

        distances = np.linalg.norm(positions - centroid, axis = 1)

        diff = distances - self.last_distances

        reward = np.where(diff < 0, self._cohesionReward, 0)

        self.last_distances = distances
        self.last_velocities = self.vel

        return reward.sum()

    def alignmentReward(self):

        return 0

    def obstacleAvoidanceReward(self):
        reward = np.zeros(self.NUM_DRONES)

        positions = self.pos

        # Ground Clearance
        groundDistance = positions[:, 2]
        reward = np.where(groundDistance < self._safeDistance, self._droneCollisionReward * np.exp(-((self._safeDistance - groundDistance) - 2)), 0)

        # Obstacle Avoidance 
        # TODO: Implement obstacle avoidance reward

        return reward.sum()
    
    def interDroneCollisionReward(self):
        reward = np.zeros(self.NUM_DRONES)

        positions = self.pos

        interDroneDistances = np.linalg.norm(positions[:, np.newaxis] - positions, axis = 2)

        reward = np.where(interDroneDistances < self._safeDistance, self._droneCollisionReward, 0)

        return reward.sum()
    
    def targetReachedReward(self):
        reward = np.zeros(self.NUM_DRONES)

        positions = self.pos

        distancesFromTarget = np.linalg.norm(positions - self.target_xyz, axis = 1)

        reward += np.where(distancesFromTarget < self._safeDistance, self._targetReachedReward * np.exp(-((self._safeDistance - distancesFromTarget) - 2)), 0)

        return reward.sum()

    def _computeReward(self):
        reward = self.cohesionReward() + self.interDroneCollisionReward() + self.obstacleAvoidanceReward() + self.targetReachedReward()
        print('[INFO] Reward:', reward)
        return reward
        
    def _computeObs(self):
        ret = super()._computeObs()
        target = np.tile(self.target_xyz, (self.NUM_DRONES, 1))

        ret = np.hstack([ret, target])
        print("[INFO] Observation: ", ret)
        return ret

    def _computeTerminated(self) -> bool:
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        dist = 0
        for i in range(self.NUM_DRONES):
            dist += np.linalg.norm(self.target_xyz[:]-states[i][0:3])
        if dist < .0001:
            return True
        else:
            return False
    
    def _computeTruncated(self) -> bool:
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > 2.0 or abs(states[i][1]) > 2.0 or states[i][2] > 2.0 # Truncate when a drones is too far away
             or abs(states[i][7]) > .4 or abs(states[i][8]) > .4 # Truncate when a drone is too tilted
            ):
                return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
    
    def _computeInfo(self):
        return {"Drone Position": self.pos, 'Num Drones': self.NUM_DRONES}
    
    def step(self, action:np.ndarray):
        print('[INFO] Action:', action)

        return super().step(action)
    
    def _computeNeighbors(self, obs):
        for i in range(0, self.NUM_DRONES):
            for j in range(0, self.NUM_DRONES):
                print('[INFO] Distance between drone', i, 'and drone', j, ':', np.linalg.norm(obs[i][0:3] - obs[j][0:3]))
                if i != j and np.linalg.norm(obs[i][0:3] - obs[j][0:3]) < .5:
                    print('[INFO] Drone', i, 'is neighbor of drone', j)

    def randomizeTargets(self):
        # randomize target position (z > _safeDistance)
        # self.target_xyz = np.random.uniform(-2, 2, size=(3))
        # self.target_xyz[2] = np.random.uniform(self._safeDistance, 2)

        self.target_xyz = np.array([0, 0, 4])

    
    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(action.shape[0]):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                    )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos
                                                        )
                rpm[k,:] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[k,:] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,target[0]])
                                                        )
                rpm[k,:] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm        

if __name__ == "__main__":  
    env = make_vec_env(MultiAgentNeighborEnv, env_kwargs=dict(num_drones=3, obs=ObservationType.KIN, act=ActionType.PID), n_envs=1, vec_env_cls=SubprocVecEnv, seed=0)

    model = PPO('MlpPolicy', env, verbose=1)

    logger = RewardLoggerCallback()
    cb = CustomCallback()

    model.learn(total_timesteps=250000, callback = cb)

    model.save("ppo_multiagent")

    logger.close_plot()




# if __name__ == "__main__":
#     # train_env = make_vec_env(MultiAgentNeighborEnv, env_kwargs=dict(num_drones=5, obs=ObservationType.KIN, act=ActionType.VEL), n_envs=1, seed=0)
#     numDrones = 5
#     startPos = np.random.uniform(-2, 2, size=(numDrones, 2))
#     startPos = np.concatenate((startPos, np.zeros((numDrones, 1))), axis=1)

#     train_env = MultiAgentNeighborEnv(num_drones=5, obs=ObservationType.KIN, act=ActionType.VEL, gui=True, initial_xyzs = startPos)

#     print('[INFO] Action space:', train_env.action_space)
#     print('[INFO] Observation space:', train_env.observation_space)

#     train_env.reset()

#     print(train_env)
#     print(type(train_env))

#     PYB_CLIENT = train_env.getPyBulletClient()
#     START = time.time()

#     for i in range(0, int(60*train_env.CTRL_FREQ)):
#         # action = np.random.rand(5, 4).astype(np.float32) - .5
#         action = np.zeros((5, 4))
#         print(action.shape, action)
#         train_env.step(action)

#         train_env.render()

#         sync(i, START, train_env.CTRL_TIMESTEP)
        
#     train_env.close()
