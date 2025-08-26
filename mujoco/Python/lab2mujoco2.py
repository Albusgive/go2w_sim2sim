import torch
import mujoco
import mujoco.viewer
import time

# 加载 mujoco 模型
m = mujoco.MjModel.from_xml_path('/home/albusgive2/go2w_sim2sim/robot/go2w_description/mjcf/scene.xml')
d = mujoco.MjData(m)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint",
    ]

isaaclab_joint_names = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint',
                        'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint',
                        'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint',
                        'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint',
                        'FL_wheel_joint', 'FR_wheel_joint', 'RL_wheel_joint',
                        'RR_wheel_joint']



def get_sensor_data(sensor_name):
    sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id == -1:
        raise ValueError(f"Sensor '{sensor_name}' not found in model!")
    start_idx = m.sensor_adr[sensor_id]
    dim = m.sensor_dim[sensor_id]
    sensor_values = d.sensordata[start_idx : start_idx + dim]
    return torch.tensor(
        sensor_values, 
        device=device, 
        dtype=torch.float32
    )


def set_joint_angle(joint_name, angle):
    joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    d.qpos[m.jnt_qposadr[joint_id]] = angle
    
    
def world2self(quat, v):
    q_w = quat[0] 
    q_vec = quat[1:] 
    v_vec = torch.tensor(v, device=device, dtype=torch.float32)
    a = v_vec * (2.0 * q_w**2 - 1.0)
    b = torch.linalg.cross(q_vec, v_vec) * q_w * 2.0
    c = q_vec * torch.dot(q_vec, v_vec) * 2.0
    result = a - b + c
    return result.to(device)


def get_obs(actions, default_dof_pos, commands=[0.0, 0.05, 0.0]):
    commands_scale = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)
    base_quat = get_sensor_data("imu_quat")
    gravity = [0.0, 0.0, -1.0]
    projected_gravity = world2self(base_quat, torch.tensor(gravity, device=device, dtype=torch.float32))
    imu_gyro = get_sensor_data("imu_gyro")
    dof_pos = torch.zeros(12, device=device, dtype=torch.float32)
    for i in range(12):
        dof_pos[i] = get_sensor_data(isaaclab_joint_names[i]+"_pos")[0]

    dof_vel = torch.zeros(16, device=device, dtype=torch.float32)
    for i in range(16):
        dof_vel[i] = get_sensor_data(isaaclab_joint_names[i]+"_vel")[0]

    cmds = torch.tensor(commands, device=device, dtype=torch.float32)

    # print("imu_gyro:", imu_gyro)
    # print("projected_gravity:", projected_gravity)
    # print("dof_pos:", dof_pos)
    # print("dof_vel:", dof_vel)
    print("commands:", cmds)
    return torch.cat(
        [
            imu_gyro * 0.25,  # 3
            projected_gravity,  # 3
            cmds * commands_scale,  # 3
            (dof_pos - default_dof_pos) * 1.0,  # 16
            dof_vel * 0.05,  # 16
            actions,  # 16
        ],
        axis=-1,
    )


def main():
    # 加载模型
    try:
        loaded_policy = torch.jit.load("/home/albusgive2/go2w_sim2sim/policy/history_1/policy.pt")
        loaded_policy.eval()  # 设置为评估模式
        loaded_policy.to(device)
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()

        
    # 初始化观察数据
    isaac_default_dof_pos_without_wheel = torch.tensor([
        0.00, 0.00, 0.00, 0.00,
        0.8, 0.8, 0.8, 0.8,
        -1.5, -1.5, -1.5, -1.5,
        ],
        device=device,
        dtype=torch.float32)
    act_default_dof_pos = torch.tensor([
        0.00, 0.80, -1.50,
        0.00, 0.80, -1.50,
        0.00, 0.80, -1.50,
        0.00, 0.80, -1.50,
        0.0, 0.0, 0.0, 0.0
        ],
        device=device,
        dtype=torch.float32)
    actions_scale = torch.tensor([
        0.125, 0.25, 0.25,
        0.125, 0.25, 0.25,
        0.125, 0.25, 0.25,
        0.125, 0.25, 0.25,
        5.0, 5.0, 5.0, 5.0], device=device, dtype=torch.float32)
    actions = torch.zeros(16, device=device, dtype=torch.float32)

    # from IPython import embed; embed()
    # 从未上电姿态站立
    # set_joint_angle("left_thigh_joint", -0.35)
    # set_joint_angle("right_thigh_joint", -0.35)
    # for i in range(200):
    #     mujoco.mj_step(m, d)
    # 启动 mujoco 渲染
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            obs = get_obs(actions=actions,
                          default_dof_pos=isaac_default_dof_pos_without_wheel)
            obs = torch.clip(obs, -100, 100)

            actions = loaded_policy(obs)

            act = actions * actions_scale + act_default_dof_pos
            act = torch.clip(act, -100, 100)
            # print("actions:", act)
            act = act.detach().cpu().numpy()
            for i in range(16):
                d.ctrl[i] = act[i]

            # 执行一步模拟
            step_start = time.time()
            for i in range(4):
                mujoco.mj_step(m, d)
            # 更新渲染
            viewer.sync()
            # 同步时间
            time_until_next_step = m.opt.timestep*4 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()