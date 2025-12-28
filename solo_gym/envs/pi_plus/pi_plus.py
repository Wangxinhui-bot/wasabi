# python
import torch

# solo-gym
from solo_gym.envs import LeggedRobot
from solo_gym import LEGGED_GYM_ROOT_DIR
from .pi_plus_config import PiPlusFlatCfg
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    torch_rand_float,
    quat_rotate,
    quat_rotate_inverse,
)
from typing import Dict
from solo_gym.utils.keyboard_controller import KeyboardAction, Delta

class PiPlus(LeggedRobot):
    cfg: PiPlusFlatCfg

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
            # runner需要motion_loader有observation_dim属性和feed_forward_generator方法
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
        # 临时保存原始的num_actions
        original_num_actions = self.num_actions
        # 临时将num_actions设置为num_dof，这样父类会初始化正确大小的p_gains和d_gains
        self.num_actions = self.num_dof
        
        # 调用父类的_init_buffers
        super()._init_buffers()
        
        # 恢复原始的num_actions
        self.num_actions = original_num_actions
        
        # 重新初始化actions和last_actions为正确的num_actions大小（24）
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # 重新初始化torques为num_dof大小，因为_compute_torques返回的是num_dof大小的张量
        self.torques = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # 注意：p_gains和d_gains现在是num_dof大小，这是正确的
        # torques也是num_dof大小，因为我们需要为所有关节设置力矩（锁定关节的力矩很小）
        # actions和last_actions是num_actions大小（24），这是正确的
        
        # 定义需要锁定的关节名称（头部和腰部）
        locked_joint_names = [
            # 头部关节 (8个)
            "neck_pitch_joint",
            "neck_roll_joint",
            "head_pitch_joint",
            "head_roll_joint",
            "head_yaw_joint",
            "l_eyelash_joint",
            "r_eyelash_joint",
            "mouth_joint",
            # 腰部关节 (3个)
            "waist_roll_joint",
            "waist_yaw_joint",
            "waist_pitch_joint",
        ]
        
        # 创建锁定关节和活动关节的索引
        self.locked_joint_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if name in locked_joint_names],
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )
        
        self.active_joint_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if name not in locked_joint_names],
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )
        
        # 为锁定关节设置PD增益，保持它们在默认位置（0）
        # 使用较大的增益值来保持位置，但不参与神经网络计算
        self.p_gains[self.locked_joint_indices] = 80.0  # 位置增益
        self.d_gains[self.locked_joint_indices] = 80.0  # 速度增益
        
        # 初始化其他缓冲区
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel], device=self.device, requires_grad=False)
        self.base_height = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        # 重新计算noise_scale_vec，因为现在active_joint_indices已经存在
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

    def compute_observations(self):
        """Computes observations - 只包含活动关节的状态"""
        # 只使用活动关节的位置和速度
        active_dof_pos = self.dof_pos[:, self.active_joint_indices]
        active_dof_vel = self.dof_vel[:, self.active_joint_indices]
        active_default_dof_pos = self.default_dof_pos[:, self.active_joint_indices]
        active_actions = self.actions
        
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, 0].unsqueeze(1) * self.commands_scale,
                (active_dof_pos - active_default_dof_pos) * self.obs_scales.dof_pos,
                active_dof_vel * self.obs_scales.dof_vel,
                active_actions,
            ),
            dim=-1,
        )
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _compute_torques(self, actions):
        """Compute torques from actions - 只对活动关节计算力矩"""
        # actions已经是23维，对应23个活动关节
        # 创建全尺寸的torques张量
        all_torques = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # 对活动关节计算力矩
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        
        if control_type == "P":
            active_torques = (
                self.p_gains[self.active_joint_indices] * 
                (actions_scaled + self.default_dof_pos[:, self.active_joint_indices] - 
                 self.dof_pos[:, self.active_joint_indices]) - 
                self.d_gains[self.active_joint_indices] * self.dof_vel[:, self.active_joint_indices]
            )
        elif control_type == "V":
            active_torques = (
                self.p_gains[self.active_joint_indices] * 
                (actions_scaled - self.dof_vel[:, self.active_joint_indices]) -
                self.d_gains[self.active_joint_indices] * 
                (self.dof_vel[:, self.active_joint_indices] - self.last_dof_vel[:, self.active_joint_indices]) / self.sim_params.dt
            )
        elif control_type == "T":
            active_torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        # 将活动关节的力矩放入全尺寸张量
        all_torques[:, self.active_joint_indices] = active_torques
        
        # 对锁定关节，使用PD控制保持默认位置（0）
        if len(self.locked_joint_indices) > 0:
            locked_default_pos = torch.zeros(
                (self.num_envs, len(self.locked_joint_indices)),
                dtype=torch.float,
                device=self.device
            )
            locked_torques = (
                self.p_gains[self.locked_joint_indices] * 
                (locked_default_pos - self.dof_pos[:, self.locked_joint_indices]) -
                self.d_gains[self.locked_joint_indices] * self.dof_vel[:, self.locked_joint_indices]
            )
            all_torques[:, self.locked_joint_indices] = locked_torques
        
        # 限制力矩范围
        return torch.clip(all_torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environments"""
        # 先调用父类方法
        super()._reset_dofs(env_ids)
        
        # 确保锁定关节保持在默认位置（0）
        if len(self.locked_joint_indices) > 0:
            self.dof_pos[env_ids][:, self.locked_joint_indices] = 0.0
            self.dof_vel[env_ids][:, self.locked_joint_indices] = 0.0
        
        # 更新仿真状态
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

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

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations"""
        # 使用配置中的num_actions作为活动关节数（23个）
        num_active_joints = self.cfg.env.num_actions
        
        # 获取obs_scales，如果不存在则从配置中获取
        if hasattr(self, 'obs_scales'):
            obs_scales = self.obs_scales
        else:
            obs_scales = self.cfg.normalization.obs_scales
        
        # 创建噪声向量，长度为观察空间维度
        noise_vec = torch.zeros(self.cfg.env.num_observations, dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        noise_vec[:3] = noise_scales.lin_vel * noise_level * obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:10] = 0.0  # commands
        noise_vec[10:10+num_active_joints] = noise_scales.dof_pos * noise_level * obs_scales.dof_pos
        noise_vec[10+num_active_joints:10+2*num_active_joints] = noise_scales.dof_vel * noise_level * obs_scales.dof_vel
        noise_vec[10+2*num_active_joints:10+3*num_active_joints] = 0.0  # previous actions
        
        return noise_vec

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

