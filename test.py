import os
import time
import torch
from stable_baselines3 import SAC
from gomoku_gym import GomokuArmEnv
import environment


environment.IS_TRAIN = False

# -----------------------------
# 配置
# -----------------------------
MODEL_PATH = "models/SAC_Gomoku/sac_gomoku_final.zip"
NUM_EPISODES = 10  # 抓取多少次
MAX_STEPS = 500    # 每次抓取包含多少个决策步
SLOW_MODE = True   # 是否慢速播放（方便观察）

# -----------------------------
# 主测试函数
# -----------------------------
def test():
    # 1. 检查模型文件
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型不存在: {MODEL_PATH}")

    print("加载环境...")
    env = GomokuArmEnv()

    print("加载模型...")
    model = SAC.load(MODEL_PATH, env=env)

    print("\n开始推理演示...\n")

    success_count = 0

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()

        print(f"\n===== Episode {episode} 开始 =====")

    
        for step in range(MAX_STEPS):
            # 2. 预测动作（无随机）
            action, _ = model.predict(obs, deterministic=True)

            # 3. 执行动作
            obs, reward, terminated, truncated, _ = env.step(action)

            # 4. 可视化控制
            if SLOW_MODE:
                time.sleep(1. / 120.)

            # 5. 成功检测
            if terminated:
                success_count += 1
                print(f"✔ Episode {episode} 成功！ step={step} reward={reward:.2f}")

            # 6. 结束条件
            if terminated or truncated:
                print(f"Episode {episode} 结束 | step={step} | reward={reward:.2f}")
                break

        else:
            # 如果没有 break（跑满 MAX_STEPS）
            print(f" Episode {episode} 超时未完成")


    print("\n==============================")
    print("测试完成")
    print(f"成功次数: {success_count}/{NUM_EPISODES}")
    print(f"成功率: {success_count / NUM_EPISODES:.2f}")
    print("==============================\n")



if __name__ == "__main__":
    try:
        test()
    except KeyboardInterrupt:
        print("\n手动中断测试")