import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
from environment import GomokuEnv

class GomokuArmEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = GomokuEnv()


        self.action_space = spaces.Box(-1, 1, (3,), dtype=np.float32)

        self.observation_space = spaces.Box(-2, 2, (10,), dtype=np.float32)

        self.max_steps = 250
        self.current_step = 0

        self.target_pos = np.zeros(3)
        self.target_indices = (0, 0)
        self.has_grasped = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.env.reset(options=options)

        self.current_step = 0
        self.has_grasped = False

        spawn = [
            np.random.uniform(-0.25, -0.1),
            np.random.uniform(0.2, 0.6),
            0.03
        ]
        self.piece_id = self.env.create_piece(
            pos=spawn,
            is_black=(self.env.current_player == 1)
        )

        if options and 'target_coord' in options:
            row, col = options['target_coord']
        else:
            empty = np.argwhere(self.env.board == 0)
            row, col = empty[np.random.choice(len(empty))]

        self.target_indices = (row, col)
        self.target_pos = np.array(
            self.env.get_physical_coord(row, col),
            dtype=np.float32
        )

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # ===== 控制机械臂 =====
        ee = p.getLinkState(self.env.robot_id, self.env.ee_index)[0]

        target = [
            ee[0] + action[0]*0.05,
            ee[1] + action[1]*0.05,
            ee[2] + action[2]*0.05
        ]

        joint = p.calculateInverseKinematics(
            self.env.robot_id,
            self.env.ee_index,
            target,
            p.getQuaternionFromEuler([np.pi,0,0])
        )

        for i in range(7):
            p.setJointMotorControl2(
                self.env.robot_id,
                i,
                p.POSITION_CONTROL,
                joint[i],
                force=200
            )

        # ===== 实时状态 =====
        piece = p.getBasePositionAndOrientation(self.piece_id)[0]
        ee = p.getLinkState(self.env.robot_id, self.env.ee_index)[0]

        dist = np.linalg.norm(np.array(ee)-np.array(piece))
        dist_target = np.linalg.norm(np.array(piece)-self.target_pos)
        dist_xy = np.linalg.norm(np.array(piece[:2])-self.target_pos[:2])

        # ===== 自动夹爪 =====
        if not self.has_grasped:
            is_open = not (dist < 0.05 and abs(ee[2]-piece[2]) < 0.03)
        else:
            is_open = not (dist_xy < 0.04 and piece[2] < 0.05)

        finger = 0.04 if is_open else 0.01
        for i in self.env.finger_indices:
            p.setJointMotorControl2(self.env.robot_id, i, p.POSITION_CONTROL, finger)

        for _ in range(5):
            self.env.step()

        # ===== 更新状态 =====
        piece = p.getBasePositionAndOrientation(self.piece_id)[0]
        ee = p.getLinkState(self.env.robot_id, self.env.ee_index)[0]

        dist = np.linalg.norm(np.array(ee)-np.array(piece))
        dist_target = np.linalg.norm(np.array(piece)-self.target_pos)
        dist_xy = np.linalg.norm(np.array(piece[:2])-self.target_pos[:2])

        obs = self._get_obs()
        reward = 0
        terminated = False

        # 抓取
        if not self.has_grasped and piece[2] > 0.05 and dist < 0.06:
            self.has_grasped = True
            reward += 50

        if not self.has_grasped:
            reward += np.exp(-10*dist)
        else:
            reward += 2
            reward += 5*np.exp(-5*dist_target)

            # 成功放置
            if is_open and dist_target < 0.04 and piece[2] < 0.05:
                r, c = self.env.snap_to_grid(self.piece_id)

                if self.env.board[r, c] != 0:
                    reward -= 50
                    terminated = True
                else:
                    self.env.board[r, c] = self.env.current_player
                    self.env.current_player = 3 - self.env.current_player
                    reward += 200
                    terminated = True

        # 掉落
        if self.has_grasped and piece[2] < 0.03:
            reward -= 20
            terminated = True

        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {
        "board": self.env.board.copy(),
        "target": self.target_indices,
        "current_player": self.env.current_player   
}

    def _get_obs(self):
        ee = p.getLinkState(self.env.robot_id, self.env.ee_index)[0]
        piece = p.getBasePositionAndOrientation(self.piece_id)[0]
        flag = [1.0] if self.has_grasped else [0.0]

        return np.array([*ee, *piece, *self.target_pos, *flag], dtype=np.float32)