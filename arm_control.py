import pybullet as p
import math
from environment import IS_TRAIN 

class ArmController:
    def __init__(self, env):
        self.env = env
        self.robot_id = env.robot_id
        self.ee_index = env.ee_index
        self.finger_indices = env.finger_indices
        self.ready_pos = env.ready_pos
        
        # 设定抓取姿态：垂直向下
        self.down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

    def move_to(self, target_pos, duration=0.1):
        actual_duration = 0.01 if IS_TRAIN else duration
        steps = max(1, int(actual_duration * 240))
        current_force = 250 if IS_TRAIN else 200

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
                    force=current_force
                )
            self.env.step()

    def control_gripper(self, open_bool):
        """夹爪控制"""
        # 0.01 为安全闭合距离，0.00 会捏碎直径0.056的棋子产生巨大斥力
        target = 0.04 if open_bool else 0.01 
        
        grip_force = 40 if not open_bool else 10
        
        for i in self.finger_indices:
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=grip_force 
            )
            
        loop_steps = 2 if IS_TRAIN else 30
        for _ in range(loop_steps): 
            self.env.step()

    def pick_piece(self, piece_pos):
        hover_pos = [piece_pos[0], piece_pos[1], piece_pos[2] + 0.15]
        grab_pos = [piece_pos[0], piece_pos[1], piece_pos[2] + 0.035]

        self.control_gripper(True)
        self.move_to(hover_pos, duration=0.8)
        self.move_to(grab_pos, duration=0.6)
        self.control_gripper(False)
        
        if not IS_TRAIN:
            for _ in range(20): self.env.step() 
        
        self.move_to(hover_pos, duration=0.6)

    def place_piece(self, target_pos):
        hover_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.15]
        drop_pos = [target_pos[0], target_pos[1], target_pos[2] + 0.06]

        self.move_to(hover_pos, duration=0.8)
        self.move_to(drop_pos, duration=0.6)
        self.control_gripper(True)
        self.move_to(hover_pos, duration=0.6)
        self.reset_to_ready()

    def reset_to_ready(self):
        reset_steps = 20 if IS_TRAIN else 100
        for _ in range(reset_steps):
            for i in range(7):
                p.setJointMotorControl2(
                    self.robot_id,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=self.ready_pos[i],
                    force=150
                )
            self.env.step()