import pybullet as p
import pybullet_data
import time
import numpy as np
import torch  

IS_TRAIN = True

class GomokuEnv:
    def __init__(self):
        use_gpu = torch.cuda.is_available()
        
        self.current_player = 1  # 1=黑 2=白
        
        if IS_TRAIN:
            self.client = p.connect(p.DIRECT)
        else:
            self.client = p.connect(p.GUI)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # ===== 单一棋盘（唯一数据源）=====
        self.board = np.zeros((15, 15), dtype=np.int8)

        # GPU 渲染
        if use_gpu and IS_TRAIN:
            try:
                import pybullet_utils.eglRenderer as egl
                p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            except:
                pass

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        # ===== 坐标系统（统一从原点开始）=====
        self.grid_size = 0.06
        self.board_size = 0.84
        self.stone_radius = 0.028

        # 棋盘（中心在正中央）
        p.loadURDF("plane.urdf")
        board_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.board_size/2, self.board_size/2, 0.01]
        )
        board_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.board_size/2, self.board_size/2, 0.01],
            rgbaColor=[0.8, 0.6, 0.4, 1]
        )
        self.board_id = p.createMultiBody(
            0, board_col, board_vis,
            [self.board_size/2, self.board_size/2, 0.01]
        )

        # ===== 机械臂 =====
        robot_start_pos = [-0.4, self.board_size/2, 0]
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            robot_start_pos,
            useFixedBase=True
        )

        self.ee_index = 11
        self.finger_indices = [9, 10]

        self.ready_pos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i in range(7):
            p.resetJointState(self.robot_id, i, self.ready_pos[i])

        self.piece_ids = []

    # ===== 逻辑 → 物理 =====
    def get_physical_coord(self, row, col):
        x = col * self.grid_size
        y = row * self.grid_size
        z = 0.03
        return [x, y, z]

    # ===== 物理 → 逻辑 + 吸附 =====
    def snap_to_grid(self, piece_id):
        pos, orn = p.getBasePositionAndOrientation(piece_id)

        col = int(round(pos[0] / self.grid_size))
        row = int(round(pos[1] / self.grid_size))

        row = int(np.clip(row, 0, 14))
        col = int(np.clip(col, 0, 14))

        new_pos = self.get_physical_coord(row, col)
        p.resetBasePositionAndOrientation(piece_id, new_pos, orn)

        return row, col

    def create_piece(self, row=None, col=None, is_black=True, pos=None):
        if pos is None:
            phys_pos = self.get_physical_coord(row, col)
        else:
            phys_pos = pos

        color = [0.1,0.1,0.1,1] if is_black else [0.9,0.9,0.9,1]

        p_col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.stone_radius)
        p_vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.stone_radius, rgbaColor=color)

        pid = p.createMultiBody(0.1, p_col, p_vis, phys_pos)

        p.changeDynamics(pid, -1, lateralFriction=2.0, rollingFriction=0.01)

        self.piece_ids.append(pid)
        return pid

    def step(self):
        p.stepSimulation()
        if not IS_TRAIN:
            time.sleep(1./240.)

    def reset(self, seed=None, options=None):
        self.board.fill(0)
        
        if options and "player" in options:
            self.current_player = options["player"]
        else:
            self.current_player = 1

        for pid in self.piece_ids:
            p.removeBody(pid)
        self.piece_ids = []

        for i in range(7):
            p.resetJointState(self.robot_id, i, self.ready_pos[i])