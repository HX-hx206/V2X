import os
import random
import numpy as np

# 每一个agent都有一个memory
class ReplayMemory:
    # 初始化：需要输入memory的容量：entry_size，初始化的代码如下：
    """
        - actions: 存储所有动作的数组。
        - rewards: 存储所有奖励的数组。
        - prestate: 存储所有状态前缀的数组。
        - poststate: 存储所有状态后缀的数组。
        - batch_size: 采样时的批量大小。
        - count: 当前记忆中存储的经验数量。
        - current: 下一个要添加的经验的索引位置。
        """
    def __init__(self, entry_size, alpha=, beta_start=, beta_frames=):
        self.entry_size = entry_size
        self.memory_size =
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.float64)
        self.prestate = np.empty((self.memory_size, self.entry_size), dtype=np.float16)
        self.poststate = np.empty((self.memory_size, self.entry_size), dtype=np.float16)
        self.priorities = np.zeros(self.memory_size, dtype=np.float32)  # 优先级数组
        self.alpha = alpha              # 优先级系数（0=均匀采样，1=完全优先级）
        self.beta = beta_start          # 重要性采样调整系数
        self.beta_frames = beta_frames  # beta线性调整到1的步数
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.current = 0
        self.count = 0
        self.batch_size = 64

    def add(self, prestate, poststate, reward, action):
        # 新样本初始优先级设为当前最大优先级或1.0
        max_prio = self.priorities.max() if self.count > 0 else 1.0
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.prestate[self.current] = prestate
        self.poststate[self.current] = poststate
        self.priorities[self.current] = max_prio  # 初始优先级
        self.current = (self.current + 1) % self.memory_size
        self.count = min(self.count + 1, self.memory_size)

    def sample(self):
        # 计算优先级概率分布
        probs = self.priorities[:self.count] ** self.alpha
        probs /= probs.sum()

        # 分段采样（替代SumTree实现）
        indices = np.random.choice(self.count, self.batch_size, p=probs)
        weights = (self.count * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化重要性采样权重

        # 更新beta值
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            self.prestate[indices],
            self.poststate[indices],
            self.actions[indices],
            self.rewards[indices],
            indices,
            weights
        )

    def update_priorities(self, indices, td_errors):
        # 用TD误差绝对值更新优先级（避免零误差）
        self.priorities[indices] = (np.abs(td_errors) + 1e-5).flatten()
