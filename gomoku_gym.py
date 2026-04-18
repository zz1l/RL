import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
from environment import GomokuEnv


class GomokuArmEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.env = GomokuEnv()

        # dx dy dz
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # ee(3) + piece(3) + target(3) + grasp_flag + gripper_flag
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(11,),
            dtype=np.float32
        )

        self.max_steps = 250
        self.current_step = 0

        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_indices = (0, 0)

        self.has_grasped = False
        self.piece_id = None

    # ======================================================
    # reset
    # ======================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.env.reset(options=options)

        self.current_step = 0
        self.has_grasped = False

        # 随机生成棋子（抓取区）
        spawn = [
            np.random.uniform(-0.20, -0.10),
            np.random.uniform(0.25, 0.55),
            0.03
        ]

        self.piece_id = self.env.create_piece(
            pos=spawn,
            is_black=(self.env.current_player == 1)
        )

        # 随机目标格子
        if options and "target_coord" in options:
            row, col = options["target_coord"]
        else:
            empty = np.argwhere(self.env.board == 0)
            row, col = empty[np.random.choice(len(empty))]

        self.target_indices = (row, col)

        self.target_pos = np.array(
            self.env.get_physical_coord(row, col),
            dtype=np.float32
        )

        obs = self._get_obs()

        return obs, {}

    # ======================================================
    # step
    # ======================================================
    def step(self, action):
        self.current_step += 1

        terminated = False
        reward = 0.0

        # ------------------------------------------
        # 当前状态（动作前）
        # ------------------------------------------
        ee = p.getLinkState(
            self.env.robot_id,
            self.env.ee_index
        )[0]

        piece = p.getBasePositionAndOrientation(
            self.piece_id
        )[0]

        prev_dist = np.linalg.norm(
            np.array(ee) - np.array(piece)
        )

        prev_target_dist = np.linalg.norm(
            np.array(piece) - self.target_pos
        )

        # ------------------------------------------
        # 控制机械臂
        # ------------------------------------------
        move_scale = 0.015

        target = [
            ee[0] + float(action[0]) * move_scale,
            ee[1] + float(action[1]) * move_scale,
            ee[2] + float(action[2]) * move_scale,
        ]

        # 防止钻地
        target[2] = max(0.01, target[2])

        joint = p.calculateInverseKinematics(
            self.env.robot_id,
            self.env.ee_index,
            target,
            p.getQuaternionFromEuler([np.pi, 0, 0])
        )

        for i in range(7):
            p.setJointMotorControl2(
                self.env.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint[i],
                force=200
            )

        # ------------------------------------------
        # 自动夹爪
        # ------------------------------------------
        if not self.has_grasped:
            # 接近棋子时闭合
            dist_xy = np.linalg.norm(
                np.array(ee[:2]) - np.array(piece[:2])
            )

            close_cond = (
                dist_xy < 0.03 and
                abs(ee[2] - piece[2]) < 0.04
            )

            is_open = not close_cond

        else:
            # 到目标点附近张开
            piece_to_target_xy = np.linalg.norm(
                np.array(piece[:2]) - self.target_pos[:2]
            )

            open_cond = (
                piece_to_target_xy < 0.04 and
                piece[2] < 0.07
            )

            is_open = open_cond

        finger = 0.04 if is_open else 0.01

        for idx in self.env.finger_indices:
            p.setJointMotorControl2(
                self.env.robot_id,
                idx,
                p.POSITION_CONTROL,
                targetPosition=finger,
                force=40
            )

        # ------------------------------------------
        # 仿真推进
        # ------------------------------------------
        for _ in range(15):
            self.env.step()

        # ------------------------------------------
        # 更新状态（动作后）
        # ------------------------------------------
        ee = p.getLinkState(
            self.env.robot_id,
            self.env.ee_index
        )[0]

        piece = p.getBasePositionAndOrientation(
            self.piece_id
        )[0]

        dist = np.linalg.norm(
            np.array(ee) - np.array(piece)
        )

        dist_target = np.linalg.norm(
            np.array(piece) - self.target_pos
        )

        dist_xy = np.linalg.norm(
            np.array(piece[:2]) - self.target_pos[:2]
        )

        # ==================================================
        # reward
        # ==================================================

        # ---------- 未抓取：靠近棋子 ----------
        if not self.has_grasped:
            reward += (prev_dist - dist) * 30.0

        # ---------- 已抓取：靠近目标 ----------
        else:
            reward += (prev_target_dist - dist_target) * 40.0

        # 时间惩罚
        reward -= 0.01

        # ---------- 抓取成功 ----------
        if (not self.has_grasped) and piece[2] > 0.035:
            self.has_grasped = True
            reward += 30.0

        # ---------- 放置成功 ----------
        if self.has_grasped:
            if is_open and dist_xy < 0.04 and piece[2] < 0.05:

                r, c = self.env.snap_to_grid(self.piece_id)

                if self.env.board[r, c] == 0:
                    self.env.board[r, c] = self.env.current_player
                    self.env.current_player = 3 - self.env.current_player

                    reward += 200.0
                else:
                    reward -= 80.0

                terminated = True

        # ---------- 掉落失败 ----------
        if self.has_grasped and piece[2] < 0.025:
            reward -= 30.0
            terminated = True

        # ---------- 超时 ----------
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()

        info = {
            "board": self.env.board.copy(),
            "target": self.target_indices,
            "current_player": self.env.current_player
        }

        return obs, reward, terminated, truncated, info

    # ======================================================
    # observation
    # ======================================================
    def _get_obs(self):
        ee = p.getLinkState(
            self.env.robot_id,
            self.env.ee_index
        )[0]

        piece = p.getBasePositionAndOrientation(
            self.piece_id
        )[0]

        grasp_flag = [1.0 if self.has_grasped else 0.0]

        finger_pos = p.getJointState(
            self.env.robot_id,
            self.env.finger_indices[0]
        )[0]

        gripper_flag = [1.0 if finger_pos > 0.03 else 0.0]

        obs = np.array(
            [
                *ee,
                *piece,
                *self.target_pos,
                *grasp_flag,
                *gripper_flag
            ],
            dtype=np.float32
        )

        return obs