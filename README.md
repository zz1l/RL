### 1.安装必要的包

~~~txt
---require.txt---

stable-baselines3==2.8.0
gymnasium==1.2.3
pybullet==3.2.7
torch==2.7.1
numpy==2.2.6
tensorboard==2.20.0
tqdm==4.67.1
matplotlib==3.10.8 
~~~



### 2. 运行说明

#### 2.1 训练

先在 `environment.py` 中将

~~~python
IS_TRAIN = False
~~~

设置为

~~~python
IS_TRAIN = True
~~~

随后运行 `train.py`

~~~python
python train.py
~~~

可以调整的参数包括

~~~bash
policy_kwargs # 网络结构
learning_rate # 学习率
buffer_size # 经验池大小，表示存多少历史经验，越大越稳定，越小训练越快
batch_size 
learning_starts # 前 learning_starts 步只收集数据，不训练, 让模型预热
tau # target network 更新速度
gamma # 未来奖励权重
train_freq # 采样步数
gradient_steps # 每一步采样训练的次数
~~~

#### 2.2 训练效果查看

#### 2.2.1 直接在控制台看细节

~~~bash
ep_len_mean # 每一次任务平均走了多少步，若为200（设置的最大步数）表示没有完成任务
ep_rew_mean # 每一局平均奖励(可以为负), 越大越好
~~~

#### 2.2.2 或者在tensorboard查看

在当前目录的终端运行：

```bash
tensorboard --logdir logs/tensorboard
```

打开浏览器：

```bash
http://localhost:6006
```

主要查看 ` rollout/ep_rew_mean` 曲线。 上升说明训练效果好

#### 2.3 推理

先在 `environment.py` 中将

~~~python
IS_TRAIN = True
~~~

设置为

~~~python
IS_TRAIN = False
~~~

随后在 `test.py`中设置

~~~python
MODEL_PATH # 选取ckp
NUM_EPISODES = 10  # 抓取多少次
MAX_STEPS = 500    # 每次抓取包含多少个决策步
SLOW_MODE = True   # 是否慢速播放（方便观察）
~~~

>**注意：`MODEL_PATH` 千万不要解压，必需是一个zip 目录**

最后运行 `train.py`



### 3. 模型优化

在 `gomoku_gym.py` 中巧妙设计 `reward` 的计算方式，并且重新训练模型



