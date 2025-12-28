from solo_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from solo_gym import LEGGED_GYM_ROOT_DIR


class PiPlus12FlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 46  # 3(lin_vel) + 3(ang_vel) + 3(gravity) + 1(command) + 12(dof_pos) + 12(dof_vel) + 12(actions) = 46
        num_actions = 12  # 每条腿3个关节(hip_pitch, hip_roll, thigh) × 4条腿 = 12

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        terrain_proportions = [0.0, 1.0]
        num_rows = 5
        max_init_terrain_level = 4

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]  # 根据pi_plus的实际高度调整
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # 前左腿 (FL) - 仅3个活动关节
            "FL_hip_pitch_joint": 0.0,
            "FL_hip_roll_joint": 0.0,
            "FL_thigh_joint": 0.0,
            # 前右腿 (FR) - 仅3个活动关节
            "FR_hip_pitch_joint": 0.0,
            "FR_hip_roll_joint": 0.0,
            "FR_thigh_joint": 0.0,
            # 后左腿 (RL) - 仅3个活动关节
            "RL_hip_pitch_joint": 0.0,
            "RL_hip_roll_joint": 0.0,
            "RL_thigh_joint": 0.0,
            # 后右腿 (RR) - 仅3个活动关节
            "RR_hip_pitch_joint": 0.0,
            "RR_hip_roll_joint": 0.0,
            "RR_thigh_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters (仅3个活动关节类型):
        stiffness = {
            'hip_pitch': 5.0,
            'hip_roll': 5.0,
            'thigh_joint': 5.0,
        }  # [N*m/rad]
        damping = {
            'hip_pitch': 0.1,
            'hip_roll': 0.1,
            'thigh_joint': 0.1,
        }  # [N*m*s/rad]
        torque_limit = 20.0  # 根据URDF中的effort限制

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_plus_34dof_251222/urdf/pi_plus_12dof.urdf'
        foot_name = "ankle_roll_link"  # 脚部链接名称（虽然锁定了，但链接仍然存在）
        terminate_after_contacts_on = ["base_link", "chest_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.85
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.35  # 根据pi_plus的实际高度调整
        max_contact_force = 350.0
        only_positive_rewards = True
        class scales(LeggedRobotCfg.rewards.scales):
            ang_vel_xy = -0.0
            base_height = -0.0
            lin_vel_z = -0.0
            orientation = -0.0
            torques = -0.000025
            feet_air_time = 0.1
            tracking_ang_vel = 0.0
            feet_contact_forces = -0.1
            ang_vel_x = -0.1
            ang_vel_z = -0.1
            lin_vel_y = -0.1

    class commands(LeggedRobotCfg.commands):
        num_commands = 1
        curriculum = False
        max_curriculum = 5.0
        resampling_time = 5.0
        heading_command = False
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.2, 0.5]

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots = True
        max_push_vel_xy = 0.5
        randomize_base_mass = True
        added_mass_range = [-0.2, 2.0]
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85

    class motion_loader:
        reference_motion_file = None  # 暂时不设置，后续可以添加
        corruption_level = 0.0
        reference_observation_horizon = 2
        test_mode = False
        test_observation_dim = None

class PiPlus12FlatCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "WASABIOnPolicyRunner"

    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [128, 128, 128]
        critic_hidden_dims = [128, 128, 128]
        init_noise_std = 1.0

    class discriminator:
        reward_coef = 0.1
        reward_lerp = 0.9  # wasabi_reward = (1 - reward_lerp) * style_reward + reward_lerp * task_reward
        style_reward_function = "wasserstein_mapping"  # log_mapping, quad_mapping, wasserstein_mapping
        shape = [512, 256]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        wasabi_replay_buffer_size = 1000000
        policy_learning_rate = 1e-3
        discriminator_learning_rate = 5e-7
        discriminator_momentum = 0.5
        discriminator_weight_decay = 1e-3
        discriminator_gradient_penalty_coef = 5
        discriminator_loss_function = "WassersteinLoss"  # MSELoss, BCEWithLogitsLoss, WassersteinLoss
        discriminator_num_mini_batches = 80

    class runner(LeggedRobotCfgPPO.runner):
        run_name = "wasabi"
        experiment_name = "flat_pi_plus_12"
        algorithm_class_name = "WASABI"
        policy_class_name = "ActorCritic"
        load_run = -1
        max_iterations = 1000
        normalize_style_reward = True
        compute_dynamic_time_warping = False
        num_dynamic_time_warping_samples = 50

