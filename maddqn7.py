from __future__ import division, print_function
import random
import scipy
import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
import Environment16
import os
from replay_memory import ReplayMemory
import matplotlib.pyplot as plt
import sys

# -------------------- 日志设置 --------------------
log_file_path = "training_log.txt"
with open(log_file_path, "w") as log_file:
    log_file.write("Training Log\n")


def log_print(message):
    print(message)
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")


log_print("CUDA is available: " + str(tf.test.is_built_with_cuda()))
log_print("CUDA runtime is available: " + str(tf.test.is_gpu_available()))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.debugging.set_log_device_placement(True)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as test_sess:
    a = tf.constant(2.0)
    b = tf.constant(3.0)
    c = tf.add(a, b)
    log_print("Test computation result: " + str(test_sess.run(c)))

tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
tf.debugging.set_log_device_placement(True)
config.gpu_options.allow_growth = True


# -------------------- 定义 Agent 类 --------------------
class Agent(object):
    def __init__(self, memory_entry_size):
        self.discount = 0.99
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)


# -------------------- 环境参数设置 --------------------
up_lanes = [i / 2.0 for i in
            [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
down_lanes = [i / 2.0 for i in
              [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
               750 - 3.5 / 2]]
left_lanes = [i / 2.0 for i in
              [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
right_lanes = [i / 2.0 for i in
               [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                1299 - 3.5 / 2]]
width = 750 / 2
height = 1298 / 2

IS_TRAIN = 1
IS_TEST = 1 - IS_TRAIN
label = 'marl_model'
n_veh =   # 参与决策车辆数量
n_neighbor =   # 每辆车考虑的邻居数量
n_RB = n_veh  # 资源块数量

env = Environment16.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game(n_veh)

# -------------------- 训练参数 --------------------
n_episode =
n_step_per_episode = int(env.time_slow / env.time_fast)  # 每个 episode 的步数
epsi_final =
epsi_anneal_length =
mini_batch_step = n_step_per_episode  # mini-batch 训练间隔
target_update_step =
num_repeats =


# -------------------- 状态获取函数 --------------------
def get_state(env, idx=(0, 0), ind_episode=1., epsi=0.02):
    # 计算 V2I 快衰
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10) / 35
    # 计算 V2V 快衰
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] -
                env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]], np.newaxis] + 10) / 35
    # 计算 V2V 干扰
    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60
    # 计算绝对信道
    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80) / 60.0
    # 计算剩余负载和剩余时间
    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference,
                           np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([idx[0]])))


# -------------------- 构建统一的计算图 --------------------
# 在这里我们使用多头注意力模块对输入进行处理，再进入后续全连接层
n_hidden_1 = 256
n_hidden_2 = 64
n_hidden_3 = 16
n_input = len(get_state(env))
n_output = n_RB * len(env.V2V_power_dB_List)  # 动作数

g = tf.Graph()
with g.as_default():
    # 定义训练标志，用于 BatchNormalization（训练时True，推理时False）
    training = tf.compat.v1.placeholder(tf.bool, name="training")

    # ---- 在线网络前向传播 ----
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input], name="state")
    # 先经过注意力模块：假设映射到 4*64=256 维，然后 reshape 成 [batch,4,64]
    d_model_attn = 64
    seq_len = 4
    attn_proj = tf.compat.v1.layers.dense(x, seq_len * d_model_attn, activation=tf.nn.relu, name="attn_proj")
    attn_input = tf.reshape(attn_proj, [-1, seq_len, d_model_attn])

    def multi_head_attention(inputs, num_heads, d_model, scope="multi_head_attention"):


    attn_output, attn_weights = multi_head_attention(attn_input, num_heads=4, d_model=d_model_attn,
                                                     scope="multi_head_attention")
    attn_output_flat = tf.reshape(attn_output, [-1, seq_len * d_model_attn])
    new_input_dim = seq_len * d_model_attn

    # 后续全连接层
    w_1 = tf.Variable(tf.compat.v1.truncated_normal([new_input_dim, n_hidden_1], stddev=0.1))
    b_1 = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_1], stddev=0.1))
    layer_1 = tf.nn.relu(tf.add(tf.matmul(attn_output_flat, w_1), b_1))
    layer_1_b = BatchNormalization()(layer_1, training=training)

    w_2 = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    b_2 = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_2], stddev=0.1))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_b, w_2), b_2))
    layer_2_b = BatchNormalization()(layer_2, training=training)

    w_3 = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    b_3 = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_3], stddev=0.1))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2_b, w_3), b_3))
    layer_3_b = BatchNormalization()(layer_3, training=training)

    w_4 = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_3, n_output], stddev=0.1))
    b_4 = tf.Variable(tf.compat.v1.truncated_normal([n_output], stddev=0.1))
    y = tf.add(tf.matmul(layer_3_b, w_4), b_4)
    g_q_action = tf.argmax(y, axis=1, name="predicted_action")

    # ---- 目标网络前向传播 ----
    x_target = tf.compat.v1.placeholder(tf.float32, [None, n_input], name="target_state")
    attn_proj_target = tf.compat.v1.layers.dense(x_target, seq_len * d_model_attn, activation=tf.nn.relu, name="attn_proj_target")
    attn_input_target = tf.reshape(attn_proj_target, [-1, seq_len, d_model_attn])
    attn_output_target, attn_weights_target = multi_head_attention(attn_input_target, num_heads=4, d_model=d_model_attn,
                                                                   scope="multi_head_attention_target")
    attn_output_flat_target = tf.reshape(attn_output_target, [-1, seq_len * d_model_attn])

    w_1_t = tf.Variable(tf.compat.v1.truncated_normal([new_input_dim, n_hidden_1], stddev=0.1))
    b_1_t = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_1], stddev=0.1))
    layer_1_t = tf.nn.relu(tf.add(tf.matmul(attn_output_flat_target, w_1_t), b_1_t))
    layer_1_t_b = BatchNormalization()(layer_1_t, training=training)

    w_2_t = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    b_2_t = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_2], stddev=0.1))
    layer_2_t = tf.nn.relu(tf.add(tf.matmul(layer_1_t_b, w_2_t), b_2_t))
    layer_2_t_b = BatchNormalization()(layer_2_t, training=training)

    w_3_t = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    b_3_t = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_3], stddev=0.1))
    layer_3_t = tf.nn.relu(tf.add(tf.matmul(layer_2_t_b, w_3_t), b_3_t))
    layer_3_t_b = BatchNormalization()(layer_3_t, training=training)

    w_4_t = tf.Variable(tf.compat.v1.truncated_normal([n_hidden_3, n_output], stddev=0.1))
    b_4_t = tf.Variable(tf.compat.v1.truncated_normal([n_output], stddev=0.1))
    y_target = tf.add(tf.matmul(layer_3_t_b, w_4_t), b_4_t)

    # 目标网络 Q 值选取
    g_target_q_idx = tf.compat.v1.placeholder(tf.int32, [None, None], name="target_q_idx")
    target_q_with_idx = tf.gather_nd(y_target, g_target_q_idx)

    # 定义损失函数与优化器
    g_target_q_t = tf.compat.v1.placeholder(tf.float32, [None], name="target_value")
    g_action = tf.compat.v1.placeholder(tf.int32, [None], name="action")
    action_one_hot = tf.one_hot(g_action, n_output, 1.0, 0.0, name="action_one_hot")
    q_acted = tf.reduce_sum(y * action_one_hot, axis=1, name="q_acted")

    g_loss = tf.reduce_mean(tf.square(g_target_q_t - q_acted), name="g_loss")
    optim = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(g_loss)

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()


# -------------------- 批量训练函数 --------------------
def q_learning_mini_batch_all(agents, current_sess):


# -------------------- 预测函数 --------------------
def predict(current_sess, s_t, ep, test_ep=False):
    n_power_levels = len(env.V2V_power_dB_List)
    if np.random.rand() < ep and not test_ep:
        pred_action = np.random.randint(n_RB * n_power_levels)
    else:
        pred_action = current_sess.run(g_q_action, feed_dict={x: [s_t], training: False})[0]
    return pred_action


# -------------------- 目标网络更新函数 --------------------
def update_target_q_network(current_sess):
    update_ops = [
        tf.compat.v1.assign(w_1_t, current_sess.run(w_1)),
        tf.compat.v1.assign(w_2_t, current_sess.run(w_2)),
        tf.compat.v1.assign(w_3_t, current_sess.run(w_3)),
        tf.compat.v1.assign(w_4_t, current_sess.run(w_4)),
        tf.compat.v1.assign(b_1_t, current_sess.run(b_1)),
        tf.compat.v1.assign(b_2_t, current_sess.run(b_2)),
        tf.compat.v1.assign(b_3_t, current_sess.run(b_3)),
        tf.compat.v1.assign(b_4_t, current_sess.run(b_4))
    ]
    current_sess.run(update_ops)


# -------------------- 初始化所有 Agent --------------------
agents = []
for ind_agent in range(n_veh * n_neighbor):
    log_print("Initializing agent " + str(ind_agent))
    agent = Agent(memory_entry_size=len(get_state(env)))
    agents.append(agent)

sess = tf.compat.v1.Session(graph=g, config=config)
sess.run(init)
log_print("Available GPUs: " + str(tf.config.experimental.list_physical_devices('GPU')))

# -------------------- 训练过程 --------------------
episode_loss_list = []
Print_sum_reward = []
Print_Avg_V2I_SINR_C = []
Print_Avg_V2V_SINR_C = []


for i_episode in range(n_episode):
    log_print("-------------------------")
    if i_episode < epsi_anneal_length:
        epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)
    else:
        epsi = epsi_final

    if i_episode % 100 == 0:
        env.renew_vehicles_positions()
        env.renew_Vehicle_neighbor()
        env.renew_vehicles_channel(3, 60, 40)
        env.renew_channels_fastfading(3, 60, 40)

    repeat_rewards = []
    repeat_V2I_SINR = []
    repeat_V2V_SINR = []
    episode_temp_loss = []  # 保存当前 episode 内所有 repeat 的 loss

    for repeat in range(num_repeats):

        for i_step in range(n_step_per_episode):
            state = get_state(env, [0, 0], i_episode / (n_episode - 1), epsi)
            action = np.random.randint(60)  # 这里随机选择动作，实际应使用 predict()函数
            next_state = get_state(env, [0, 0], i_episode / (n_episode - 1), epsi)
            reward_val = np.random.rand()  # dummy reward
            for agent in agents:
                agent.memory.add(state, next_state, reward_val, action)
            sum_reward += reward_val
            # 每隔一定步数进行 mini-batch 训练
            if i_episode > :
                if i_step % mini_batch_step == mini_batch_step - 1:
                    loss_val_batch = q_learning_mini_batch_all(agents, sess)
                    repeat_loss.append(loss_val_batch)
        if len(repeat_loss) > 0:
            avg_repeat_loss = np.mean(repeat_loss)
            episode_temp_loss.append(avg_repeat_loss)
        repeat_rewards.append(sum_reward)
        repeat_V2I_SINR.append(current_V2I_SINR)
        repeat_V2V_SINR.append(current_V2V_SINR)
        log_print(f"Episode: {i_episode}, Repeat: {repeat}, Reward: {round(sum_reward, 4)}, "
                  f"Avg V2I SINR: {round(current_V2I_SINR, 4)}, Avg V2V SINR: {round(current_V2V_SINR, 4)}")
    avg_episode_loss = np.mean(episode_temp_loss) if episode_temp_loss else 0
    episode_loss_list.append(avg_episode_loss)
    Print_sum_reward.append(np.mean(repeat_rewards))
    Print_Avg_V2I_SINR_C.append(np.mean(repeat_V2I_SINR))
    Print_Avg_V2V_SINR_C.append(np.mean(repeat_V2V_SINR))
    # if i_episode % 10 == 0:
    #     dqn_model.update_target_network()

# -------------------- 绘制图形 --------------------
# 绘制每个 episode 的平均 loss 曲线图

sess.close()
