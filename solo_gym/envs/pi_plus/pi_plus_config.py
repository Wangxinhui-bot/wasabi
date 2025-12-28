from solo_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from solo_gym import LEGGED_GYM_ROOT_DIR


class PiPlusFlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 82  # 3(lin_vel) + 3(ang_vel) + 3(gravity) + 1(command) + 24(dof_pos) + 24(dof_vel) + 24(actions) = 82
        num_actions = 24  # 35个revolute关节 - 8个头部 - 3个腰部 = 24

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
            # 前左腿 (FL)
            "FL_hip_pitch_joint": 0.0,
            "FL_hip_roll_joint": 0.0,
            "FL_thigh_joint": 0.0,
            "FL_calf_joint": -1.5,
            "FL_ankle_pitch_joint": 1.5,
            "FL_ankle_roll_joint": 0.0,
            # 前右腿 (FR)
            "FR_hip_pitch_joint": 0.0,
            "FR_hip_roll_joint": 0.0,
            "FR_thigh_joint": 0.0,
            "FR_calf_joint": -1.5,
            "FR_ankle_pitch_joint": 1.5,
            "FR_ankle_roll_joint": 0.0,
            # 后左腿 (RL)
            "RL_hip_pitch_joint": 0.0,
            "RL_hip_roll_joint": 0.0,
            "RL_thigh_joint": 0.0,
            "RL_calf_joint": -1.5,
            "RL_ankle_pitch_joint": 1.5,
            "RL_ankle_roll_joint": 0.0,
            # 后右腿 (RR)
            "RR_hip_pitch_joint": 0.0,
            "RR_hip_roll_joint": 0.0,
            "RR_thigh_joint": 0.0,
            "RR_calf_joint": -1.5,
            "RR_ankle_pitch_joint": 1.5,
            "RR_ankle_roll_joint": 0.0,
            # 头部关节 (锁定，不参与神经网络计算)
            "neck_pitch_joint": 0.0,
            "neck_roll_joint": 0.0,
            "head_pitch_joint": 0.0,
            "head_roll_joint": 0.0,
            "head_yaw_joint": 0.0,
            "l_eyelash_joint": 0.0,
            "r_eyelash_joint": 0.0,
            "mouth_joint": 0.0,
            # 腰部关节 (锁定，不参与神经网络计算)
            "waist_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_pitch_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters (仅腿部关节):
        # 键名需要能匹配关节名称中的子串
        stiffness = {
            'hip_pitch': 5.0,
            'hip_roll': 5.0,
            'thigh_joint': 5.0,
            'calf_joint': 5.0,
            'ankle_pitch': 5.0,
            'ankle_roll': 5.0,
        }  # [N*m/rad]
        damping = {
            'hip_pitch': 0.1,
            'hip_roll': 0.1,
            'thigh_joint': 0.1,
            'calf_joint': 0.1,
            'ankle_pitch': 0.1,
            'ankle_roll': 0.1,
        }  # [N*m*s/rad]
        torque_limit = 20.0  # 根据URDF中的effort限制

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_plus_34dof_251222/urdf/pi_plus_34dof_251222.urdf'
        foot_name = "ankle_roll_link"  # 脚部链接名称
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

class PiPlusFlatCfgPPO(LeggedRobotCfgPPO):
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
        experiment_name = "flat_pi_plus"
        algorithm_class_name = "WASABI"
        policy_class_name = "ActorCritic"
        load_run = -1
        max_iterations = 1000
        normalize_style_reward = True
        compute_dynamic_time_warping = False
        num_dynamic_time_warping_samples = 50

