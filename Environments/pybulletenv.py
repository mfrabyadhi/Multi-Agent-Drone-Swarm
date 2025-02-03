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
        self.reward_hist = np.concatenate((self.reward_hist, self.locals['rewards']), axis = 0)

        # if(self.num_timesteps % self.save_interval == 0):
        np.savetxt("Reward Multi.csv", self.reward_hist, delimiter = ',')
        np.savetxt("Drone Position Multi.csv", self.drone_pos_hist, delimiter = ',')

class MultiAgentNeighborEnv(BaseRLAviary):
    """
        Multi-agent environment for neighbor-based cooperative control of multiple drones.

    """
    
    def __init__(self, drone_model = DroneModel.CF2X, num_drones = 1, neighbourhood_radius = np.inf, initial_xyzs=None, initial_rpys=None, physics = Physics.PYB, pyb_freq = 240, ctrl_freq = 240, gui=False, record=False, obs = ObservationType.KIN, act = ActionType.RPM, neighbor_selection = 'closest'):
        super().__init__(drone_model, num_drones, neighbourhood_radius, initial_xyzs, initial_rpys, physics, pyb_freq, ctrl_freq, gui, record, obs, act)

        self.neighbor_selection = neighbor_selection

        self.last_distances = np.array([0 for _ in range(num_drones)])

        self._cohesionReward = 1

        self._droneCollisionReward = -100

        self._distanceFromObstacleReward = -2

        self._distanceFromTargetReward = 3

        self._targetReachedReward = 5

        self._alignmentToTargetReward = 2

        self._timeSpentReward = -1

        self._safeDistance = .1

        self.EPISODE_LEN_SEC = 60

        self.target_xyz = np.array([0, 0, 0])

        self.stringout = ""

        self.onair = np.zeros(self.NUM_DRONES)

    def _observationSpace(self):
        """
        Returns a Box of shape (NUM_DRONES, 10), corresponding to:
        [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, target_x, target_y, target_z]
        for each drone.
        """
        lo = -np.inf
        hi =  np.inf

        # We have 10 dimensions per drone (3 pos, 4 quat, 3 target)
        # Create (NUM_DRONES, 10) lower and upper bounds
        obs_lower_bound = np.tile(
            np.array([lo] * 10, dtype=np.float32),
            (self.NUM_DRONES, 1)
        )
        obs_upper_bound = np.tile(
            np.array([hi] * 10, dtype=np.float32),
            (self.NUM_DRONES, 1)
        )

        # Create the final Box
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

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
        reward = np.where(groundDistance < self._safeDistance, self._distanceFromObstacleReward * np.exp((self._safeDistance - 5 * groundDistance)), 0)
        reward = np.where(groundDistance < self._safeDistance / 2, 4 * self._distanceFromObstacleReward, 0)


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
        reward = 0

        positions = self.pos
        velocities = self.vel

        distancesFromTarget = np.linalg.norm(positions - self.target_xyz, axis = 1, keepdims=True)


        if (self.NUM_DRONES > 1):
            anglesFromTarget = np.zeros((self.NUM_DRONES, 1))
            
            for i in range(self.NUM_DRONES):
                anglesFromTarget[i] =  np.dot(self.normalize(velocities[i]), self.normalize(self.target_xyz - positions[i]))
        else:
            anglesFromTarget = np.dot(self.normalize(velocities), self.normalize(self.target_xyz - positions).T )


        # self.stringout += " Distance: " + str(distancesFromTarget) + " Angle: " + str(anglesFromTarget)

        # reward += np.where(distancesFromTarget < self._safeDistance, self._targetReachedReward * np.exp((self._safeDistance - 5 * distancesFromTarget)), 0)
        reward += np.where(distancesFromTarget < self._safeDistance, self._targetReachedReward * np.exp((self._safeDistance - 5 * distancesFromTarget)), 0).sum()
        reward += self._alignmentToTargetReward * anglesFromTarget.sum()

        return reward.sum()
    
    def timeSpentReward(self):
        centroid = np.mean(self.pos, axis = 0)

        return np.where(np.linalg.norm(self.target_xyz - centroid) < self._safeDistance, 0, self._timeSpentReward)

    def _computeReward(self):
        reward = 0 
        if (self.NUM_DRONES > 1):
            reward = self.cohesionReward() + self.interDroneCollisionReward()
            self.stringout += "Cohesion Reward: " + str(self.cohesionReward()) + " Inter-Drone Collision Reward: " + str(self.interDroneCollisionReward())

        reward += self.obstacleAvoidanceReward() + self.targetReachedReward()
        self.stringout += " Obstacle Avoidance Reward: " + str(self.obstacleAvoidanceReward()) + " Target Reached Reward: " + str(self.targetReachedReward())
        # print(stringout)
        return reward
    
    def _getAllPosQuatAndTarget(self):
        """
        Returns a (NUM_DRONES, 10)-shaped array containing:
            - Drone position (3D) 
            - Drone orientation quaternion (4D)
            - Target position (3D) replicated for each drone

        Output row for drone i:
            [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, target_x, target_y, target_z]
        """
        # Replicate target_pos along the rows for each drone
        targets = np.tile(self.target_xyz, (self.NUM_DRONES, 1))  # (NUM_DRONES, 3)


        # Concatenate along axis=1:
        #   first 3 columns: pos
        #   next 4 columns: quat
        #   last 3 columns: target
        all_data = np.concatenate([self.pos, self.quat, targets], axis=1)

        # Cast to float32, if desired
        return all_data.astype(np.float32)
        
    def _computeObs(self):
        ret = self._getAllPosQuatAndTarget()

        self.stringout += f" Observation: {ret[0, :3]}, {ret[0, 3:7]}, {ret[0, 7:]}"
        # print("[INFO] Observation: ", ret)
        return ret

    def _computeTerminated(self) -> bool:
        position = self.pos
        for i in range(self.NUM_DRONES):
            if np.linalg.norm(position[i] - self.target_xyz) < self._safeDistance / 2:
                return True

        return False
    
    def _computeTruncated(self) -> bool:
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > 2.0 or abs(states[i][1]) > 2.0 or states[i][2] > 2.0 # Truncate when a drones is too far away
             or abs(states[i][7]) > .4 or abs(states[i][8]) > .4 # Truncate when a drone is too tilted
            ):
                return True
            
            if (self.onair[i] and states[i][2] < self._safeDistance):
                return True
            
        

        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
        return False

    def _computeInfo(self):
        # print(self.stringout)
        self.stringout = ""
        return {"Drone Position": self.pos, 'Num Drones': self.NUM_DRONES}
    
    def step(self, action:np.ndarray):
        self.stringout = " Action: " + str(action)
        obs, reward, terminated, truncated, info = super().step(action)

        self.onair = np.where(self.onair == 0, np.where(self.pos[:, 2] > self._safeDistance, 1, 0), 1)

        return obs, reward, terminated, truncated, info
    
    def _computeNeighbors(self, obs):
        for i in range(0, self.NUM_DRONES):
            for j in range(0, self.NUM_DRONES):
                print('[INFO] Distance between drone', i, 'and drone', j, ':', np.linalg.norm(obs[i][0:3] - obs[j][0:3]))
                if i != j and np.linalg.norm(obs[i][0:3] - obs[j][0:3]) < .5:
                    print('[INFO] Drone', i, 'is neighbor of drone', j)

    def reset(self, seed=None, options=None):
        self.INIT_XYZS = np.vstack([np.array([x*4*self.L+np.random.uniform(-1,1,1) for x in range(self.NUM_DRONES)]), \
                                        np.array([y*4*self.L+np.random.uniform(-1,1,1) for y in range(self.NUM_DRONES)]), \
                                        np.ones(self.NUM_DRONES) * (self.COLLISION_H/2-self.COLLISION_Z_OFFSET+.1)]).transpose().reshape(self.NUM_DRONES, 3)

        obs, info = super().reset(seed, options)

        print("[INFO] Starting Position: ", self.pos)

        self.randomizeTargets()

        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 1, 1])
        sphere_id = p.createMultiBody(baseVisualShapeIndex=sphere_visual, basePosition=self.target_xyz)

        return obs, info

    def randomizeTargets(self):
        # randomize target position (z > _safeDistance)
        self.target_xyz = np.random.uniform(-3, 3, size=(3))
        self.target_xyz[2] = np.random.uniform(self._safeDistance, 3)

        print("[INFO] Target Position: ", self.target_xyz)

        # self.target_xyz = np.array([0, 0, 4]) 

    
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

    def normalize(self, vec):
        if len(vec.shape) == 1:
            length = np.linalg.norm(vec)
            if length == 0:
                length = .0001
            return vec / length

        length = np.linalg.norm(vec, axis = 1, keepdims=True)
        length[length == 0] = .0001

        return vec / length

if __name__ == "__main__":
    # env = make_vec_env(MultiAgentNeighborEnv, env_kwargs=dict(num_drones=1, obs=ObservationType.KIN, act=ActionType.VEL), n_envs=1, seed=0)
    env = make_vec_env(MultiAgentNeighborEnv, env_kwargs=dict(num_drones=1, obs=ObservationType.KIN, act=ActionType.VEL), n_envs=10, vec_env_cls=SubprocVecEnv, seed=0)

    # model = PPO('MlpPolicy', env, verbose=1)
    # # model = PPO.load("ppo_multiagent", env = env)

    # cb = PlotRewardCallback()

    # model.learn(total_timesteps=1000000, callback = cb)

    # model.save("ppo_multiagent")

    test_model = PPO.load("ppo_multiagent")

    test_env = MultiAgentNeighborEnv(gui = True, num_drones=1, obs=ObservationType.KIN, act=ActionType.VEL)

    mean_reward, std_reward = evaluate_policy(test_model, test_env, n_eval_episodes=10)

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = test_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)

        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)

        if terminated:
            obs = test_env.reset(seed=42, options={})

    test_env.close()

# leader-follower
# 2 mode
# no need for rotational control