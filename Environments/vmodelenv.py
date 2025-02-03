import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

import keyboard

import itertools
from collections import deque


class MultiDroneEnv(gym.Env):
    def __init__(self, n_drones: int = 1, max_speed: float = 1.0, drone_radius: float = 0.2, world_size: float = 100.0,
        dt: float = 0.01, max_steps: int = 200, min_zoom: float = 2.0, margin: float = 1.0 ):
        '''
        Multi Drone Environment 
        n_drones: number of drones
        max_speed: maximum speed of drones
        drone_radius: radius of drones
        world_size: size of the world
        dt: time step
        max_steps: maximum number of steps
        '''
        super(MultiDroneEnv, self).__init__()

        self.n_drones = n_drones
        self.max_speed = max_speed
        self.drone_radius = drone_radius
        self.collision_distance = 2.0 * drone_radius
        self.world_size = world_size
        self.dt = dt
        self.max_steps = max_steps
        
        # For the dynamic camera (zoom):
        self.min_zoom = min_zoom
        self.margin = margin

        self.agents = self.possible_agents = [f"drone_{i}" for i in range(self.n_drones)]

        # Action space (dict): Each drone has a 2D velocity command
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Observation space (dict): positions (n_drones x 2), velocities (n_drones x 2)
        self.observation_space = spaces.Dict({
            "positions": spaces.Box(
                low=-self.world_size,
                high=self.world_size,
                shape=(2,),
                dtype=np.float32
            ),
            "velocities": spaces.Box(
                low=-self.max_speed,
                high=self.max_speed,
                shape=(2,),
                dtype=np.float32
            ),
            "neighbors_vector": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32
            ),
            "target": spaces.Box(
                low=-self.world_size,
                high=self.world_size,
                shape=(2,),
                dtype=np.float32
            )
        })

        # Internal state
        self.positions = None
        self.velocities = None
        self.neighbors_vectors = None
        self.target = None
        self.step_count = 0

        # For storing history
        self.positions_history = []
        self.velocities_history = []

        # Figure handles for rendering
        self.fig = None
        self.ax = None
        self.tails = queue.Queue(maxsize=5)

        if self.render_mode != None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))

    def reset(self, seed=None, options=None):
        """
        Reset environment state and return initial observation.
        """
        super().reset(seed=seed)
        self.step_count = 0
        
        # Random initial positions (center-ish, to reduce collision chances at start)
        spawn_pos = ((self.world_size / 4 if self.world_size > 10 else 2.0) if self.world_size < 4 else 2.0)
        self.positions = np.random.uniform(low = -spawn_pos, high = spawn_pos, size = (self.n_drones, 2))

        # Start velocities at 0 or randomize if desired
        self.velocities = np.zeros_like(self.positions)

        # Clear histories
        self.positions_history = [self.positions.copy()]
        self.velocities_history = [self.velocities.copy()]

        return self._computeObs(), {}

    def step(self, action):
        """
        Step the environment by one time step.
        action: dictionary {"drone_0": (vx, vy), ...}
        """
        self.step_count += 1

        raw_velocities = np.array(tuple(action.values()))
        raw_velocities_norm = np.linalg.norm(raw_velocities, axis=1)
        raw_velocities_norm[raw_velocities_norm == 0] = .1
        
        self.velocities = raw_velocities / raw_velocities_norm[:, None] * self.max_speed
            
        self.positions += self.velocities * self.dt

        self.positions_history.append(self.positions.copy())
        self.velocities_history.append(self.positions.copy())

        # Simple reward
        reward = self._computeReward()
        
        # Check collisions
        done = self._computeTerminated()

        truncated = self._computeTruncated()

        obs = self._computeObs()
        info = {}
        
        return obs, reward, done, truncated, info

    def render(self):
        """
        Render the drones on a 2D plot using a zoom-based camera.
        1. Compute center as the average position of all drones.
        2. Find the max distance of any drone from that center, add margin + radius.
        3. Ensure we don't zoom below min_zoom.
        """
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            print("Creating new figure for rendering. {}".format(self.fig))

        self.ax.clear()

        # Camera center is the average of drone positions
        camera_center = np.mean(self.positions, axis=0)
        
        # Distances from center
        distances = np.linalg.norm(self.positions - camera_center, axis=1)
        max_dist = np.max(distances) if len(distances) > 0 else 0.0

        # Define camera range as the max distance plus drone radius + margin
        camera_range = max_dist + self.drone_radius + self.margin

        # Enforce minimum zoom
        camera_range = max(camera_range, self.min_zoom)
        
        # We make the view a square:
        x_min = camera_center[0] - camera_range
        x_max = camera_center[0] + camera_range
        y_min = camera_center[1] - camera_range
        y_max = camera_center[1] + camera_range

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect('equal', 'box')

        # Plot each drone
        for i in range(self.n_drones):
            x, y = self.positions[i]
            vx, vy = self.velocities[i] / np.linalg.norm(self.velocities[i])

            # Drone circle
            circle = plt.Circle((x, y), self.drone_radius, 
                                color='blue', fill=True, alpha=0.6)
            self.ax.add_patch(circle)

            # Velocity arrow
            self.ax.arrow(
                x + self.drone_radius * vx, y + self.drone_radius * vy,
                vx * 0.5, vy * 0.5,  # scale arrow length if desired
                head_width=0.05, head_length=0.08,
                fc='red', ec='red'
            )

            print("=====================================")

            if self.step_count % 5 == 0:
            #     Draw 5 line from previous position to current position
                self.tails.put((self.positions_history[-2][i], self.positions_history[-1][i]))

            # if self.step_count % 25 == 0:
            #     tails = np.array(list(self.tails.queue))
            #     self.ax.plot(tails[:, :, 0].T, tails[:, :, 1].T, color='black', alpha=0.5)
                
            print("+++++++++++++++++++++++++++++++++++++")

        self.draw_debug_range()

        self.ax.set_title(f"Step: {self.step_count}")


        plt.pause(self.dt)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None

    def plot_trajectories(self):
        """
        Plots the stored positions and velocities over time for each drone.
        In this method, we plot the entire path at once; we might simply
        fit all data so you can see every position in a fixed view.
        """
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_aspect('equal', 'box')

        pos_hist = np.array(self.positions_history)  # shape (t, n, 2)
        vel_hist = np.array(self.velocities_history) # shape (t, n, 2)

        # Determine bounding box from all data
        all_x = pos_hist[:, :, 0].flatten()
        all_y = pos_hist[:, :, 1].flatten()
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        
        margin = self.margin
        x_min -= (self.drone_radius + margin)
        x_max += (self.drone_radius + margin)
        y_min -= (self.drone_radius + margin)
        y_max += (self.drone_radius + margin)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Plot each drone's trajectory
        for i in range(self.n_drones):
            x_vals = pos_hist[:, i, 0]
            y_vals = pos_hist[:, i, 1]
            
            ax.plot(x_vals, y_vals, label=f"Drone {i} Traj")

        #     # Optionally draw velocity arrows every few steps
        #     for t in range(len(x_vals)):
        #         if t % 5 == 0:  # arrow every 5 steps
        #             vx = vel_hist[t, i, 0]
        #             vy = vel_hist[t, i, 1]
        #             ax.arrow(
        #                 x_vals[t], y_vals[t],
        #                 vx * 0.3, vy * 0.3,
        #                 head_width=0.05, head_length=0.07,
        #                 fc='red', ec='red', alpha=0.5
        #             )

        ax.legend()
        ax.set_title("Drones Trajectories Over Time")
        plt.show()

    # -------------------------------
    # Helper Methods
    # -------------------------------
    def _computeObs(self):
        return {agent: self._get_agent_obs(agent) for agent in self.agents}
    
    def _get_agent_obs(self, agent):
        index = int(agent.split("_")[1])
        agent_obs = {}
        for obs_key in self.observation_space.keys():
            if obs_key == "positions":
                agent_obs[obs_key] = self.positions[index]
            elif obs_key == "velocities":
                agent_obs[obs_key] = self.velocities[index]
            elif obs_key == "neighbors_vector":
                agent_obs[obs_key] = self._get_neighbors_vector(index)
            elif obs_key == "target":
                agent_obs[obs_key] = self.target

        return agent_obs

    def _get_neighbors_vector(self, index):
        return (0, 0)
    
    def _computeReward(self):
        return 0.0
    
    def _computeTerminated(self):
        return self._check_collisions()
    
    def _computeTruncated(self):
        return False

    def _check_collisions(self):
        """
        Checks if any pair of drones is within self.collision_distance.
        Returns True if collision is found, else False.
        """
        distances = np.linalg.norm(np.abs(self.positions - self.positions[:, None]))

        ret = bool((distances < self.collision_distance).any())

        if ret:
            print("Collision detected!")

        return ret
    
    def draw_circle(self, center, radius, color='blue', fill=True, alpha=0.6):
        circle = plt.Circle(center, radius, color=color, fill=fill, alpha=alpha)
        self.ax.add_patch(circle)

    def draw_debug_range(self):
        referencedrone = 0
        center = self.positions[referencedrone]
        radius = self.collision_distance

        self.draw_circle(center, radius, color='red', fill=False, alpha=0.6)

        for i in range(self.n_drones):
            if i == referencedrone:
                continue
            center = self.positions[i]
            radius = self.collision_distance
            self.draw_circle(center, radius, color='red', fill=False, alpha=0.6)

            distance = np.linalg.norm(self.positions[referencedrone] - self.positions[i])

            color = 'red' if distance < self.collision_distance else 'blue'

            plt.plot([self.positions[referencedrone][0], self.positions[i][0]], [self.positions[referencedrone][1], self.positions[i][1]], color=color, linestyle='--')
            
            plt.text((self.positions[referencedrone][0] + self.positions[i][0]) / 2, (self.positions[referencedrone][1] + self.positions[i][1]) / 2, f"{distance:.2f}", color=color)


# ------------
# Example usage
# ------------
if __name__ == "__main__":
    # env = make_vec_env(MultiAgentNeighborEnv, env_kwargs=dict(num_drones=3, obs=ObservationType.KIN, act=ActionType.PID), n_envs=1, vec_env_cls=SubprocVecEnv, seed=0)
    
    env = MultiDroneEnv(n_drones=3, max_speed = 5.0)
    obs, info = env.reset()

    radius = 50
    drone_center = np.vstack((env.positions[:, 0], env.positions[:, 1])).T
    angle = np.full_like(drone_center[:, 0], np.pi / 2)


    up_action = (0, 1)
    down_action = (0, -1)
    left_action = (-1, 0)
    right_action = (1, 0)
    actions = { f"drone_{i}": (0, 0) for i in range(env.n_drones) }

    for _ in itertools.count(start = 1):
        # Sample random actions from action space
        action = [0, 0]
        
        keyboard.is_pressed("a")
        keys = keyboard._pressed_events.keys()
        
        if(len(keys) > 0):
            print(keys)

        if 16 in keys:
            break

        if 72 in keys:
            action[1] += 1
        if 80 in keys:
            action[1] -= 1
        if 75 in keys:
            action[0] -= 1
        if 77 in keys:
            action[0] += 1


        # vx = np.sin(angle) * -radius
        # vy = np.cos(angle) * radius

        # action = { f"drone_{i}" : (vx[i], vy[i]) for i in range(env.n_drones) }
        # action = {
        #     f"drone_{i}": env.action_space[f"drone_{i}"].sample()
        #     for i in range(env.n_drones)
        # }

        action = np.clip(action, -1.0, 1.0)
        actions["drone_0"] = action

        # print(action)

        obs, reward, done, truncated, info = env.step(actions)

        # print(obs)

        env.render()

        relative_positions = env.positions - drone_center

        # print(relative_positions)

        angle = np.arctan2(relative_positions[:, 1], relative_positions[:, 0])
        
        if truncated:
            print("Episode truncated.")
            break

        if done:
            print("Episode finished due to collision or max steps.")
            break

    env.plot_trajectories()
    env.close()
