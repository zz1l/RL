import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# 从 gomoku_gym 导入环境
from gomoku_gym import GomokuArmEnv  

def train():
    env = GomokuArmEnv()

    model_path = "models/SAC_Gomoku"
    log_path = "logs/tensorboard"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    model = SAC(
      "MlpPolicy",
      env,
      policy_kwargs=dict(net_arch=[512, 512, 512]),
      verbose=1,
      device="cuda" if torch.cuda.is_available() else "cpu",
      tensorboard_log=log_path,
      learning_rate=3e-4,
      buffer_size=1000000,
      batch_size=2048,
      learning_starts=10000,
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

    print(f"正在使用 {'GPU' if torch.cuda.is_available() else 'CPU'} 进行训练...")
    try:
        model.learn(
            total_timesteps=500000,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n手动停止训练，正在保存当前模型...")
    finally:
        model.save(f"{model_path}/sac_gomoku_final")
        print(f"训练完成！模型保存在: {model_path}")

if __name__ == "__main__":
    train()