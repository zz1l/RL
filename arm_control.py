import pybullet as p
import math
import numpy as np
from environment import IS_TRAIN 

class ArmController:
    def __init__(self, env):
        self.env = env
        self.robot_id = env.robot_id
        self.ee_index = env.ee_index
        self.finger_indices = env.finger_indices
        self.ready_pos = env.ready_pos

        # 垂直向下抓取
        self.down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

    # 基础控制 

    def move_to(self, target_pos, duration=0.1):
        actual_duration = 0.01 if IS_TRAIN else duration
        steps = max(1, int(actual_duration * 240))
        force = 250 if IS_TRAIN else 200

        for _ in range(steps):
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_index,
                target_pos,
                self.down_orn,
                maxNumIterations=100,
                residualThreshold=1e-5
            )

            for i in range(7):
                p.setJointMotorControl2(
                    self.robot_id,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=joint_poses[i],
                    force=force
                )
            self.env.step()

    def control_gripper(self, open_bool):
        target = 0.04 if open_bool else 0.01
        force = 40 if not open_bool else 10

        for i in self.finger_indices:
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=force
            )

        steps = 2 if IS_TRAIN else 30
        for _ in range(steps):
            self.env.step()

    # 核心动作

    def pick_piece(self, piece_id):
        piece_pos = p.getBasePositionAndOrientation(piece_id)[0]

        hover = [piece_pos[0], piece_pos[1], piece_pos[2] + 0.15]
        grab = [piece_pos[0], piece_pos[1], piece_pos[2] + 0.035]

        self.control_gripper(True)
        self.move_to(hover, 0.8)
        self.move_to(grab, 0.6)
        self.control_gripper(False)

        if not IS_TRAIN:
            for _ in range(20):
                self.env.step()

        self.move_to(hover, 0.6)

    def place_piece(self, piece_id, target_pos):
        hover = [target_pos[0], target_pos[1], target_pos[2] + 0.15]
        drop = [target_pos[0], target_pos[1], target_pos[2] + 0.06]

        self.move_to(hover, 0.8)
        self.move_to(drop, 0.6)

        self.control_gripper(True)

        # 吸附 + 棋盘同步 
        row, col = self.env.snap_to_grid(piece_id)

        if self.env.board[row, col] != 0:
            print(f"[警告] 非法落子: ({row}, {col}) 已有棋子")
        else:
            self.env.board[row, col] = 1

        self.move_to(hover, 0.6)
        self.reset_to_ready()

        return row, col

    def reset_to_ready(self):
        steps = 20 if IS_TRAIN else 100

        for _ in range(steps):
            for i in range(7):
                p.setJointMotorControl2(
                    self.robot_id,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=self.ready_pos[i],
                    force=150
                )
            self.env.step()

    # 系统级接口

    def execute_move(self, target_coord):
        """
        外部调用接口（AI / 人类）
        输入: (row, col)
        输出: 更新后的棋盘
        """

        row, col = target_coord

        # 1. 生成棋子（在抓取区）
        spawn = [
            np.random.uniform(-0.25, -0.1),
            np.random.uniform(0.2, 0.6),
            0.03
        ]
        piece_id = self.env.create_piece(pos=spawn)

        # 2. 计算目标物理位置
        target_pos = self.env.get_physical_coord(row, col)

        # 3. 执行抓取
        self.pick_piece(piece_id)

        # 4. 执行放置
        final_row, final_col = self.place_piece(piece_id, target_pos)

        return self.env.board.copy()