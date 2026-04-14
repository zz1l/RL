import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
from environment import GomokuEnv

class GomokuArmEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = GomokuEnv()

        self.action_space = spaces.Box(-1,1,(4,),dtype=np.float32)
        self.observation_space = spaces.Box(-2,2,(7,),dtype=np.float32)

        self.max_steps = 200
        self.current_step = 0

    def clamp(self, pos):
        return [
            np.clip(pos[0], -0.5, 0.8),
            np.clip(pos[1], 0.0, 0.9),
            np.clip(pos[2], 0.02, 0.5)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        self.current_step = 0

        spawn = [
            np.random.uniform(-0.25, -0.1),
            np.random.uniform(0.2, 0.6),
            0.03
        ]

        self.piece_id = self.env.create_piece(pos=spawn)
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # 当前末端位置
        current_pos = p.getLinkState(self.env.robot_id, self.env.ee_index)[0]

        target_pos = [
            current_pos[0] + action[0]*0.05,
            current_pos[1] + action[1]*0.05,
            current_pos[2] + action[2]*0.05
        ]

        target_pos = self.clamp(target_pos)

        # 固定朝下
        orn = p.getQuaternionFromEuler([np.pi,0,0])

        joint = p.calculateInverseKinematics(
            self.env.robot_id,
            self.env.ee_index,
            target_pos,
            orn
        )

        for i in range(7):
            p.setJointMotorControl2(
                self.env.robot_id,
                i,
                p.POSITION_CONTROL,
                joint[i],
                force=200
            )

        # 夹爪
        is_open = action[3] > 0
        finger = 0.04 if is_open else 0.01
        for i in self.env.finger_indices:
            p.setJointMotorControl2(self.env.robot_id, i, p.POSITION_CONTROL, finger, force=40)

        # 多步执行
        for _ in range(5):
            self.env.step()

        obs = self._get_obs()

        ee = obs[0:3]
        piece = obs[3:6]

        dist = np.linalg.norm(ee - piece)

        # -------- reward --------
        reward = -dist * 2.0

        terminated = False
        # 接近奖励
        if dist < 0.15: reward += 1
        if dist < 0.08: reward += 2
        if dist < 0.05: reward += 5
        
        # 张开夹爪（接近阶段）
        if dist < 0.1 and is_open:
            reward += 0.5
        
        # ⭐ 新增：闭合夹爪（抓取阶段）
        if dist < 0.05 and not is_open:
            reward += 5.0
        
        # ⭐ 抓住提升
        if dist < 0.05 and piece[2] > 0.05:
            reward += 20
        
        # ⭐ 抬起连续奖励
        reward += piece[2] * 10
        
        # 动作惩罚
        reward -= 0.01 * np.linalg.norm(action)
        
        # 成功
        if piece[2] > 0.12 and dist < 0.1:
            reward += 100
            terminated = True

        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        ee = p.getLinkState(self.env.robot_id, self.env.ee_index)[0]
        piece = p.getBasePositionAndOrientation(self.piece_id)[0]
        grip = p.getJointState(self.env.robot_id, self.env.finger_indices[0])[0]

        return np.array([*ee, *piece, grip], dtype=np.float32)