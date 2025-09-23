import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import *
import numpy as np
import copy
import random


class ContrastiveLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=64):
        super(ContrastiveLearner, self).__init__()
        self.output_dim = output_dim
        # encoder backbone
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # projection head g: 2-layer MLP
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
        # predictor head h: 2-layer MLP
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

    @staticmethod
    def _cosine_similarity(p: torch.Tensor, z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return (p * z).sum(dim=1)

    def simsiam_loss_from_states(self, states: torch.Tensor, noise_std: float = 0.02, drop_prob: float = 0.1, mask_ratio: float = 0.05) -> torch.Tensor:
        """Compute SimSiam loss from raw states by generating two augmented views.
        Args:
            states: [B, state_dim]
            noise_std: Gaussian noise std for view-2
            drop_prob: feature dropout prob applied independently per feature
            mask_ratio: fraction of features to mask to zero
        Returns:
            scalar loss tensor
        """
        if states.dim() == 1:
            states = states.unsqueeze(0)
        B = states.size(0)
        # view1: feature dropout + masking
        if drop_prob > 0:
            keep = (torch.rand_like(states) > drop_prob).float()
            v1 = states * keep
        else:
            v1 = states
        if mask_ratio > 0:
            k = max(1, int(states.size(1) * mask_ratio))
            mask_idx = torch.randint(0, states.size(1), (B, k), device=states.device)
            v1 = v1.clone()
            v1[torch.arange(B).unsqueeze(1), mask_idx] = 0.0
        # view2: light Gaussian noise
        noise = torch.randn_like(states) * noise_std
        v2 = states + noise

        # encode -> project
        z1 = self.projector(self.forward(v1))
        z2 = self.projector(self.forward(v2))
        # predict
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        # stop-grad on opposite branch (SimSiam)
        with torch.no_grad():
            z1_sg = z1.detach()
            z2_sg = z2.detach()
        loss = 0.5 * (
            1.0 - self._cosine_similarity(p1, z2_sg).mean() +
            1.0 - self._cosine_similarity(p2, z1_sg).mean()
        )
        return loss


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc_mu = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.fc_logstd = nn.Linear(args.rnn_hidden_dim, args.action_dim)

        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc_mu, gain=0.01)
            orthogonal_init(self.fc_logstd, gain=0.01)

    def forward(self, actor_input, avail_a_n):
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        mu = torch.tanh(self.fc_mu(self.rnn_hidden))
        logstd = self.fc_logstd(self.rnn_hidden)
        std = F.softplus(logstd) + 1e-5
        std = torch.clamp(std, 1e-3, 1.0)

        return mu, std


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc_mu = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.fc_logstd = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc_mu, gain=0.01)
            orthogonal_init(self.fc_logstd, gain=0.01)

    def forward(self, actor_input, avail_a_n):
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        logstd = self.fc_logstd(x)
        std = F.softplus(logstd) + 1e-5
        std = torch.clamp(std, 1e-3, 1.0)
        return mu, std


class Critic(nn.Module):
    def __init__(self, args, feature_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(feature_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, features):
        x = self.activate_func(self.fc1(features))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class MAPPO:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.contrastive_lambda = args.contrastive_lambda
        self.max_train_steps = args.max_train_steps
        self.critic_coef = args.critic_coef
        self.contrastive_update_freq = args.contrastive_update_freq
        self.min_entropy_coef = args.min_entropy_coef
        self.stop_contrastive_after_ratio = args.stop_contrastive_after_ratio

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_agent_specific = args.use_agent_specific
        self.use_value_clip = args.use_value_clip

        self.actor_input_dim = args.obs_dim

        self.contrastive_learner = ContrastiveLearner(self.state_dim)
        self.critic = Critic(args, feature_dim=self.contrastive_learner.output_dim)

        if self.use_rnn:
            self.actor = Actor_RNN(args, self.actor_input_dim)
        else:
            self.actor = Actor_MLP(args, self.actor_input_dim)

        if self.set_adam_eps:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
            # SimSiam branch: AdamW with higher lr and weight decay
            self.contrastive_optimizer = torch.optim.AdamW(self.contrastive_learner.parameters(), lr=2e-3, weight_decay=1e-4, eps=1e-5)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
            self.contrastive_optimizer = torch.optim.AdamW(self.contrastive_learner.parameters(), lr=2e-3, weight_decay=1e-4)

    def choose_action(self, obs_n, avail_a_n, evaluate):
        with torch.no_grad():
            if evaluate:
                self.actor.eval()
            else:
                self.actor.train()

            actor_inputs = []
            obs_n = torch.tensor(obs_n, dtype=torch.float32)
            actor_inputs.append(obs_n)
            if self.add_agent_id:
                actor_inputs.append(torch.eye(self.N))
            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)
            avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)
            mu, std = self.actor(actor_inputs, avail_a_n)
            if evaluate:
                a_n = mu
                return a_n.numpy(), None
            else:
                dist = Normal(mu, std)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n).sum(dim=-1)
                return a_n.numpy(), a_logprob_n.numpy()

    def get_value(self, s, obs_n):
        with torch.no_grad():
            # 估值时，所有网络都应在评估模式
            self.contrastive_learner.eval()
            self.critic.eval()
            
            s_tensor = torch.tensor(s, dtype=torch.float32)
            if len(s_tensor.shape) == 1:
                s_tensor = s_tensor.unsqueeze(0)

            features = self.contrastive_learner(s_tensor)
            value = self.critic(features)
            
            # 中心化Critic，所有智能体共享同一个价值
            v_n = value.repeat(1, self.N)
            return v_n.numpy().flatten()

    def lr_decay(self, total_steps):
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_now
        for p in self.contrastive_optimizer.param_groups:
            p['lr'] = lr_now
    
    def get_inputs(self, batch, max_episode_len):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])
        # Critic的输入是全局状态 s
        critic_inputs = batch['s']

        if self.add_agent_id:
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, max_episode_len, 1, 1)
            actor_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat(actor_inputs, dim=-1)
        return actor_inputs, critic_inputs

    def train(self, replay_buffer, total_steps):
        # 确保网络在训练模式
        self.critic.train()
        self.contrastive_learner.train()
        
        batch = replay_buffer.get_training_data()
        max_episode_len = replay_buffer.max_episode_len
        
        contrastive_loss_epoch = []

        # GAE
        with torch.no_grad():
            # v_n in buffer has shape [B, T+1, N], r has shape [B, T, N]
            deltas = batch['r'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['dw']) - batch['v_n'][:, :-1]
            adv = []
            gae = 0
            for t in reversed(range(max_episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * (1 - batch['dw'][:, t]) * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)
            v_target = adv + batch['v_n'][:, :max_episode_len]
            if self.use_adv_norm:
                adv_copy = adv.clone().numpy()
                adv_copy[batch['active'].numpy() == 0] = np.nan
                adv = (adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5)

        actor_inputs, critic_inputs = self.get_inputs(batch, max_episode_len)

        progress = total_steps / self.max_train_steps
        entropy_coef_now = max(self.min_entropy_coef, self.entropy_coef * (1.0 - progress))

        # 训练 Actor 和 Critic
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                # Actor forward
                if self.use_rnn:
                    self.actor.rnn_hidden = None
                    mu_list, std_list = [], []
                    for t in range(max_episode_len):
                        mu, std = self.actor(actor_inputs[index, t].reshape(self.mini_batch_size * self.N, -1),
                                             batch['avail_a_n'][index, t].reshape(self.mini_batch_size * self.N, -1))
                        mu_list.append(mu.reshape(self.mini_batch_size, self.N, -1))
                        std_list.append(std.reshape(self.mini_batch_size, self.N, -1))
                    mu_now = torch.stack(mu_list, dim=1)
                    std_now = torch.stack(std_list, dim=1)
                else:
                    mu_now, std_now = self.actor(
                        actor_inputs[index].reshape(-1, self.actor_input_dim),
                        batch['avail_a_n'][index].reshape(-1, self.action_dim)
                    )
                    mu_now = mu_now.reshape(self.mini_batch_size, max_episode_len, self.N, -1)
                    std_now = std_now.reshape(self.mini_batch_size, max_episode_len, self.N, -1)

                dist_now = Normal(mu_now, std_now)
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index]).sum(dim=-1)  # [B', T, N]
                entropy_now = dist_now.entropy().sum(dim=-1)  # [B', T, N]

                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss_per = -torch.min(surr1, surr2)  # [B', T, N]

                mask = batch['active'][index]
                mask_sum = mask.sum().clamp_min(1.0)
                actor_loss = (actor_loss_per * mask).sum() / mask_sum
                entropy_term = (entropy_now * mask).sum() / mask_sum

                # Critic Loss
                with torch.no_grad():
                    features = self.contrastive_learner(critic_inputs[index].reshape(-1, self.state_dim))
                values_now = self.critic(features.detach())
                values_now = values_now.reshape(self.mini_batch_size, max_episode_len, 1).repeat(1, 1, self.N)

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :max_episode_len].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss_per = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss_per = (values_now - v_target[index]) ** 2
                critic_loss = (critic_loss_per * mask).sum() / mask_sum
                
                # Update Actor
                self.actor_optimizer.zero_grad()
                (actor_loss - entropy_coef_now * entropy_term).backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.actor_optimizer.step()

                # Update Critic
                self.critic_optimizer.zero_grad()
                (critic_loss * self.critic_coef).backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
                self.critic_optimizer.step()

        # --- Completely Separate Update for Contrastive Learner ---
        # 每次都训练 SimSiam 分支（与 Actor/Critic 完全分离）
        self.contrastive_learner.train()
        batch_for_cl = replay_buffer.get_training_data()
        _, critic_inputs_cl = self.get_inputs(batch_for_cl, max_episode_len)
        active_mask = batch_for_cl['active'].sum(dim=-1) > 0  # [B, T]
        if active_mask.any():
            states_all = critic_inputs_cl[active_mask]  # [N_active, state_dim]
            # 子采样，避免过大计算；提升多样性
            max_samples = 2048
            if states_all.size(0) > max_samples:
                idx = torch.randperm(states_all.size(0))[:max_samples]
                states_all = states_all[idx]
            if states_all.size(0) > 1:
                contrastive_loss = self.contrastive_learner.simsiam_loss_from_states(states_all, noise_std=0.02, drop_prob=0.1, mask_ratio=0.05)
                self.contrastive_optimizer.zero_grad()
                (contrastive_loss * self.contrastive_lambda).backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.contrastive_learner.parameters(), 10.0)
                self.contrastive_optimizer.step()
                contrastive_loss_epoch.append(contrastive_loss.item())

        if self.use_lr_decay:
            self.lr_decay(total_steps)

        return np.mean(contrastive_loss_epoch) if contrastive_loss_epoch else 0

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.actor.state_dict(),
                   f"./model/MAPPO_actor_env_{env_name}_number_{number}_seed_{seed}_step_{int(total_steps / 1000)}k.pth")
        torch.save(self.critic.state_dict(),
                   f"./model/MAPPO_critic_env_{env_name}_number_{number}_seed_{seed}_step_{int(total_steps / 1000)}k.pth")
        torch.save(self.contrastive_learner.state_dict(),
                   f"./model/MAPPO_contrastive_env_{env_name}_number_{number}_seed_{seed}_step_{int(total_steps / 1000)}k.pth")

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(torch.load(
            f"./model/MAPPO_actor_env_{env_name}_number_{number}_seed_{seed}_step_{step}k.pth"))
        self.critic.load_state_dict(torch.load(
            f"./model/MAPPO_critic_env_{env_name}_number_{number}_seed_{seed}_step_{step}k.pth"))
        self.contrastive_learner.load_state_dict(torch.load(
            f"./model/MAPPO_contrastive_env_{env_name}_number_{number}_seed_{seed}_step_{step}k.pth"))
