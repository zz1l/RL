import pybullet as p
import pybullet_data
import time
import os
import torch  
import pkgutil

IS_TRAIN = True

class GomokuEnv:
    def __init__(self):
        # 1. 检查并配置 GPU 加速
        use_gpu = torch.cuda.is_available()
        
        # 连接 GUI
        if IS_TRAIN:
            self.client = p.connect(p.DIRECT, options="--opengl2")
        else:
            self.client = p.connect(p.GUI, options="--opengl2")
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        
        # 2. GPU EGL 加速（仅训练模式启用）
        if use_gpu and IS_TRAIN:
            print("尝试加载 EGL 硬件加速...")
        
            try:
                import pybullet_utils.eglRenderer as egl
                self.plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                print("EGL 插件加载成功")
                print("plugin id:", self.plugin)
                
            except Exception as e:
                print(f"EGL 加载失败 : {e}")

        # 3. GUI 高级优化设置
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        if IS_TRAIN:
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

        # 4. 物理尺寸设定 (巨型演示级比例)
        self.grid_size = 0.06     # 网格间距: 60mm
        self.board_size = 0.84    # 14个间隔，棋盘对弈区总宽: 840mm
        self.stone_radius = 0.028 # 棋子半径: 28mm 

        # 5. 建模棋盘 (严格交点对齐)
        p.loadURDF("plane.urdf")
        physical_board_size = self.board_size + 0.08 
        board_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[physical_board_size/2, physical_board_size/2, 0.01])
        board_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[physical_board_size/2, physical_board_size/2, 0.01], rgbaColor=[0.8, 0.6, 0.4, 1])
        self.board_id = p.createMultiBody(0, board_col, board_vis, [self.board_size/2, self.board_size/2, 0.01])

        # 绘制网格线 (仅在非训练模式下绘制，节省性能)
        if not IS_TRAIN:
            line_color = [0, 0, 0]
            for i in range(15):
                offset = i * self.grid_size
                p.addUserDebugLine([offset, 0, 0.021], [offset, self.board_size, 0.021], line_color, lineWidth=2)
                p.addUserDebugLine([0, offset, 0.021], [self.board_size, offset, 0.021], line_color, lineWidth=2)

        # 6. 导入机械臂
        robot_start_pos = [-0.4, self.board_size/2, 0]
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", robot_start_pos, [0, 0, 0, 1], useFixedBase=True)
        self.ee_index = 11
        self.finger_indices = [9, 10]

        self.ready_pos = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i in range(7):
            p.resetJointState(self.robot_id, i, self.ready_pos[i])
            
        self.piece_ids = []
        p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=-45, cameraPitch=-30, cameraTargetPosition=[self.board_size/2, self.board_size/2, 0])

    def get_physical_coord(self, row, col):
        """逻辑交点坐标 -> 物理空间 (X, Y, Z)"""
        x = col * self.grid_size
        y = row * self.grid_size
        z = 0.03
        return [x, y, z]

    def create_piece(self, row=None, col=None, is_black=True, pos=None):
        """
        双模态接口：
        - 人类下棋/指定落子点：传入 row, col
        - RL训练随机生成待抓取目标：传入 pos (如 [-0.2, 0.4, 0.03])
        """
        if pos is None:
            if row is None or col is None:
                raise ValueError("必须提供 (row, col) 或 pos")
            phys_pos = self.get_physical_coord(row, col)
        else:
            phys_pos = pos
            
        color = [0.1, 0.1, 0.1, 1] if is_black else [0.9, 0.9, 0.9, 1]
        p_col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.stone_radius)
        p_vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.stone_radius, rgbaColor=color)
        pid = p.createMultiBody(0.1, p_col, p_vis, phys_pos)
        
        # 必须加入 rollingFriction 防止球体一直滚动导致状态无法收敛
        p.changeDynamics(pid, -1, lateralFriction=2.0, rollingFriction=0.01)
        self.piece_ids.append(pid)
        return pid

    def step(self):
        p.stepSimulation()
        if not IS_TRAIN: 
            time.sleep(1./240.)

    def reset(self):
        # 1. 移除棋子
        for pid in self.piece_ids:
            p.removeBody(pid)
        self.piece_ids = []

        # 2. 重置关节位置与速度
        for i in range(p.getNumJoints(self.robot_id)):
            p.resetJointState(self.robot_id, i, 0) # 先归零
        
        for i in range(7):
            p.resetJointState(self.robot_id, i, self.ready_pos[i])
        
        # 3. 清除末端和基座的残余速度
        p.resetBaseVelocity(self.robot_id, [0,0,0], [0,0,0])

# --- 人类交互解析工具 ---
def human_coord_to_logic(move_str):
    move_str = move_str.strip().upper()
    if len(move_str) < 2 or len(move_str) > 3: return None
    col_char, row_str = move_str[0], move_str[1:]
    if not ('A' <= col_char <= 'O'): return None
    col = ord(col_char) - ord('A')
    try:
        row = int(row_str) - 1
        if not (0 <= row <= 14): return None
    except ValueError: return None
    return row, col

if __name__ == "__main__":
    env = GomokuEnv()
    print("\n[交互就绪] 请在终端输入落子坐标 (如: H8, K10)，输入 quit 退出。")
    while True:
        user_input = input("你的落子 -> ")
        if user_input.lower() == 'quit': break
        result = human_coord_to_logic(user_input)
        if result:
            row, col = result
            # 调用接口模拟人类下棋
            env.create_piece(row=row, col=col, is_black=False) 
            print(f">> 识别成功: 在交点 [{row}, {col}] 生成白子。")
        else:
            print("!! 格式错误: 请输入 A-O 字母加 1-15 数字组合 (如 H8)。")
        for _ in range(50): env.step()