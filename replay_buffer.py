import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {
            'obs_n': np.zeros([self.batch_size, self.episode_limit, self.N, self.obs_dim], dtype=np.float32),
            's': np.zeros([self.batch_size, self.episode_limit, self.state_dim], dtype=np.float32),
            'v_n': np.zeros([self.batch_size, self.episode_limit + 1, self.N], dtype=np.float32),
            'avail_a_n': np.ones([self.batch_size, self.episode_limit, self.N, self.action_dim], dtype=np.float32),
            'a_n': np.zeros([self.batch_size, self.episode_limit, self.N, self.action_dim], dtype=np.float32),
            'a_logprob_n': np.zeros([self.batch_size, self.episode_limit, self.N], dtype=np.float32),
            'r': np.zeros([self.batch_size, self.episode_limit, self.N], dtype=np.float32),
            'dw': np.ones([self.batch_size, self.episode_limit, self.N], dtype=np.float32),
            'active': np.zeros([self.batch_size, self.episode_limit, self.N], dtype=np.float32),
            'ep_rewards': np.zeros([self.batch_size], dtype=np.float32),
            'adv': np.zeros([self.batch_size, self.episode_limit, self.N], dtype=np.float32)
        }
        self.episode_num = 0
        self.max_episode_len = 0

    def store_transition(self, episode_step, obs_n, s, v_n, avail_a_n, a_n, a_logprob_n, r, dw):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
        self.buffer['r'][self.episode_num][episode_step] = np.array(r, dtype=np.float32).repeat(self.N)
        self.buffer['dw'][self.episode_num][episode_step] = np.array(dw, dtype=np.float32).repeat(self.N)
        self.buffer['active'][self.episode_num][episode_step] = np.ones(self.N, dtype=np.float32)

        # 修复：确保r是标量后再累加
        scalar_r = r.item() if isinstance(r, np.ndarray) else float(r)
        self.buffer['ep_rewards'][self.episode_num] += scalar_r

    def store_last_value(self, episode_step, v_n):
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.episode_num += 1
        if episode_step > self.max_episode_len:
            self.max_episode_len = episode_step

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'v_n':
                batch[key] = torch.tensor(self.buffer[key][:, :self.max_episode_len + 1], dtype=torch.float32)
            elif key in ['ep_rewards']:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
            else:
                batch[key] = torch.tensor(self.buffer[key][:, :self.max_episode_len], dtype=torch.float32)
        return batch
