from inference_api import ai_predict
from gomoku_gym import GomokuArmEnv
from arm_control import ArmController

env = GomokuArmEnv()
controller = ArmController(env.env)

# ===== 玩家选择 =====
human_player = int(input("选择执子 (1=黑, 2=白): "))
ai_player = 3 - human_player

current_player = 1  # 黑先

while True:

    # ===== AI回合 =====
    if current_player == ai_player:
        result = ai_predict(
            env.env.board.tolist(),
            difficulty="medium",
            current_player=ai_player
        )
        move = result["move"]
        print(f"AI落子: {move}")

    # ===== 人类回合 =====
    else:
        row = int(input("row: "))
        col = int(input("col: "))
        move = (row, col)

    # ===== 执行机械臂 =====
    obs, _ = env.reset(options={"target_coord": move})

    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        if done:
            break

    # ===== 更新棋盘 =====
    env.env.board[move[0]][move[1]] = current_player

    # ===== 切换玩家 =====
    current_player = 3 - current_player