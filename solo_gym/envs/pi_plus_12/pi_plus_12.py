# python
import torch

# solo-gym
from solo_gym.envs import LeggedRobot
from solo_gym import LEGGED_GYM_ROOT_DIR
from .pi_plus_12_config import PiPlus12FlatCfg
from isaacgym import gymapi
from isaacgym.torch_utils import (
    torch_rand_float,
    quat_rotate_inverse,
)
from typing import Dict
from solo_gym.utils.keyboard_controller import KeyboardAction, Delta

class PiPlus12(LeggedRobot):
    cfg: PiPlus12FlatCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # load AMP components (if motion file is provided)
        from learning.datasets.motion_loader import MotionLoader
        self.reference_motion_file = self.cfg.motion_loader.reference_motion_file
        self.test_mode = self.cfg.motion_loader.test_mode
        self.test_observation_dim = self.cfg.motion_loader.test_observation_dim
        self.reference_observation_horizon = self.cfg.motion_loader.reference_observation_horizon
        if self.reference_motion_file is not None:
            self.motion_loader = MotionLoader(
                device=self.device,
                motion_file=self.reference_motion_file,
                corruption_level=self.cfg.motion_loader.corruption_level,
                reference_observation_horizon=self.reference_observation_horizon,
                test_mode=self.test_mode,
                test_observation_dim=self.test_observation_dim
            )
            self.reference_state_idx_dict = self.motion_loader.state_idx_dict
            self.reference_full_dim = sum([ids[1] - ids[0] for ids in self.reference_state_idx_dict.values()])
            self.reference_observation_dim = sum([ids[1] - ids[0] for state, ids in self.reference_state_idx_dict.items() if ((state != "base_pos") and (state != "base_quat"))])
            self.wasabi_states = torch.zeros(
                self.num_envs, self.reference_full_dim, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.wasabi_observation_buf = torch.zeros(
                self.num_envs, self.reference_observation_horizon, self.reference_observation_dim, dtype=torch.float, device=self.device, requires_grad=False
            )
        else:
            # 如果没有motion文件，创建一个占位符motion_loader对象
            class PlaceholderMotionLoader:
                def __init__(self, observation_dim, device, reference_observation_horizon=2):
                    self.observation_dim = observation_dim
                    self.state_idx_dict = {}
                    self.device = device
                    self.reference_observation_horizon = reference_observation_horizon
                    self.observation_start_dim = 0
                    # 创建空的预加载状态，用于feed_forward_generator
                    self.num_preload_transitions = 1000
                    self.preloaded_states = torch.zeros(
                        self.num_preload_transitions,
                        self.reference_observation_horizon,
                        self.observation_dim,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False
                    )
                
                def feed_forward_generator(self, num_mini_batch, mini_batch_size):
                    """生成器方法，返回空的批次数据"""
                    for _ in range(num_mini_batch):
                        ids = torch.randint(0, self.num_preload_transitions, (mini_batch_size,), device=self.device)
                        states = self.preloaded_states[ids, :, self.observation_start_dim:]
                        yield states
            
            # 使用配置中的test_observation_dim，或者使用默认值
            placeholder_obs_dim = self.test_observation_dim if self.test_observation_dim is not None else self.cfg.env.num_observations
            self.motion_loader = PlaceholderMotionLoader(
                placeholder_obs_dim, 
                self.device, 
                self.reference_observation_horizon
            )
            self.reference_state_idx_dict = {}
            self.reference_full_dim = 0
            self.reference_observation_dim = placeholder_obs_dim
            self.wasabi_states = torch.zeros(
                self.num_envs, 0, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.wasabi_observation_buf = torch.zeros(
                self.num_envs, self.reference_observation_horizon, self.reference_observation_dim, dtype=torch.float, device=self.device, requires_grad=False
            )
        self.discriminator = None  # assigned in runner
        self.wasabi_state_normalizer = None  # assigned in runner
        self.wasabi_style_reward_normalizer = None  # assigned in runner
        if self.reference_motion_file is not None:
            self.wasabi_observation_buf[:, -1] = self.get_wasabi_observations()

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.cfg.asset.enable_joint_force_sensors:
            self.gym.refresh_dof_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_height[:] = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1, keepdim=True)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        if self.reference_motion_file is not None:
            self.wasabi_record_states()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        if self.reference_motion_file is not None:
            self.update_wasabi_observation_buf()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def update_wasabi_observation_buf(self):
        self.wasabi_observation_buf[:, :-1] = self.wasabi_observation_buf[:, 1:].clone()
        self.wasabi_observation_buf[:, -1] = self.next_wasabi_observations.clone()

    def get_wasabi_observation_buf(self):
        return self.wasabi_observation_buf.clone()

    def get_wasabi_observations(self):
        if self.test_mode:
            wasabi_obs = torch.zeros(self.num_envs, self.test_observation_dim, device=self.device, requires_grad=False)
        elif self.reference_motion_file is not None:
            wasabi_obs = self.wasabi_states[:, self.motion_loader.observation_start_dim:].clone()
        else:
            # 如果没有motion文件，返回观察空间的副本
            wasabi_obs = self.obs_buf.clone()
        return wasabi_obs

    def wasabi_record_states(self):
        if self.reference_motion_file is not None:
            for key, value in self.reference_state_idx_dict.items():
                if key == "base_pos":
                    self.wasabi_states[:, value[0]: value[1]] = self._get_base_pos()
                elif key == "feet_pos":
                    self.wasabi_states[:, value[0]: value[1]] = self._get_feet_pos()
                else:
                    self.wasabi_states[:, value[0]: value[1]] = getattr(self, key)

    def _get_base_pos(self):
        return self.root_states[:, :3] - self.env_origins[:, :3]

    def _get_feet_pos(self):
        feet_pos_global = self.rigid_body_pos[:, self.feet_indices, :3]
        feet_pos_local = torch.zeros_like(feet_pos_global)
        for i in range(len(self.feet_indices)):
            feet_pos_local[:, i] = quat_rotate_inverse(
                self.base_quat,
                feet_pos_global[:, i]
            )
        return feet_pos_local.flatten(1, 2)

    def _init_buffers(self):
        """Initialize buffers - override to add base_height"""
        super()._init_buffers()
        # Initialize base_height buffer
        self.base_height = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel], device=self.device, requires_grad=False)

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.discriminator is not None and self.wasabi_state_normalizer is not None and self.reference_motion_file is not None:
            self.next_wasabi_observations = self.get_wasabi_observations()
            wasabi_observation_buf = torch.cat((self.wasabi_observation_buf[:, 1:], self.next_wasabi_observations.unsqueeze(1)), dim=1)
            task_rew = self.rew_buf
            tot_rew, style_rew = self.discriminator.predict_wasabi_reward(wasabi_observation_buf, task_rew, self.dt, self.wasabi_state_normalizer, self.wasabi_style_reward_normalizer)
            self.episode_sums["task"] += task_rew
            self.episode_sums["style"] += style_rew
            self.rew_buf = tot_rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        if hasattr(self, 'reference_motion_file') and self.reference_motion_file is not None:
            self.episode_sums["task"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.episode_sums["style"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

    def _get_keyboard_events(self) -> Dict[str, KeyboardAction]:
        """Simple keyboard controller for linear and angular velocity."""

        def print_command():
            print("[LeggedRobot]: Environment 0 command: ", self.commands[0])

        key_board_events = {
            "u": Delta("lin_vel_x", amount=0.1, variable_reference=self.commands[:, 0], callback=print_command),
            "j": Delta("lin_vel_x", amount=-0.1, variable_reference=self.commands[:, 0], callback=print_command),
        }
        return key_board_events

    def _reward_ang_vel_x(self):
        return torch.abs(self.base_ang_vel[:, 0])

    def _reward_lin_vel_y(self):
        return torch.abs(self.base_lin_vel[:, 1])

    def _reward_ang_vel_z(self):
        return torch.abs(self.base_ang_vel[:, 2])

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        # reward only on first contact with the ground
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (only x axis)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :1] - self.base_lin_vel[:, :1]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

