import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gomoku_gym import GomokuArmEnv  

def make_env():
    env = GomokuArmEnv()
    env = Monitor(env) 
    return env

def train():
    # ===== 向量化 + 归一化 =====
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model_path = "models/SAC_Gomoku"
    log_path = "logs/tensorboard"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=log_path,
        learning_rate=3e-4,
        buffer_size=500000,
        batch_size=256,
        learning_starts=5000,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=4
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=model_path,
        name_prefix="sac_model"
    )

    print(f"使用 {'GPU' if torch.cuda.is_available() else 'CPU'} 训练")

    try:
        model.learn(
            total_timesteps=500000,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("手动停止训练")

    # 保存模型 + 归一化参数
    model.save(f"{model_path}/sac_gomoku_final")
    env.save(f"{model_path}/vec_normalize.pkl")

    print("训练完成")

if __name__ == "__main__":
    train()