import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from sim_mappo import MAPPO
from env import SecurityDetectionEnv
from collections import deque


class Runner_MAPPO:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env = SecurityDetectionEnv(
            num_terminals=15,
            num_edges=3,
            num_clouds=1
        )
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]
        self.args.obs_dim = self.env_info["obs_shape"]
        self.args.state_dim = self.env_info["state_shape"]
        self.args.action_dim = self.env_info["action_dim"]
        self.args.episode_limit = self.env_info["episode_limit"]
        self.args.contrastive_lambda = 0.2

        print("env_info['n_agents'] =", self.env_info["n_agents"])
        print("args.N after assignment =", self.args.N)

        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        self.agent_n = MAPPO(self.args)
        self.replay_buffer = ReplayBuffer(self.args)
        self.writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))
        self.win_rates = []
        self.total_steps = 0
        # Deques for logging
        self.training_raw_rewards = deque(maxlen=100)
        self.contrastive_losses = deque(maxlen=100)

        if self.args.use_reward_norm:
            self.reward_norm = Normalization(shape=1)
        elif self.args.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num = -1
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1

            # Collect experience
            raw_rew, _, episode_steps, _, _, _, _ = self.run_episode(evaluate=False)
            if raw_rew is not None:
                self.training_raw_rewards.append(raw_rew)

            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                contrastive_loss = self.agent_n.train(self.replay_buffer, self.total_steps)
                if contrastive_loss > 0:
                    self.contrastive_losses.append(contrastive_loss)
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        win_times = 0
        evaluate_reward = 0
        e_avg_delay = 0
        e_acc = 0
        e_bi = 0
        e_mu = 0
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _, avg_delay, acc,bi,mu = self.run_episode(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward
            e_avg_delay += avg_delay
            e_acc += acc
            e_bi += bi
            e_mu += mu

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        e_avg_delay = e_avg_delay / self.args.evaluate_times
        e_acc = e_acc / self.args.evaluate_times
        e_bi = e_bi/self.args.evaluate_times
        e_mu = e_mu / self.args.evaluate_times
        self.win_rates.append(win_rate)

        avg_contrastive_loss = np.mean(self.contrastive_losses) if self.contrastive_losses else 0

        print("total_steps:{} \t evaluate_reward:{:.4f} \t contrastive_loss:{:.4f} \t average_delay:{:.4f} \t accuracy:{:.4f} \t bi:{} \t mu:{}".format(
            self.total_steps,
            evaluate_reward,
            avg_contrastive_loss,
            e_avg_delay,
            e_acc,
            e_bi,
            e_mu
        ))

        # Log evaluation metrics
        self.writer.add_scalar('reward/evaluation_raw_reward', evaluate_reward, global_step=self.total_steps)
        self.writer.add_scalar('performance/evaluation_average_delay', e_avg_delay, global_step=self.total_steps)
        self.writer.add_scalar('performance/evaluation_accuracy', e_acc, global_step=self.total_steps)

        # Log training metrics (sliding window average)
        if self.training_raw_rewards:
            avg_training_raw_reward = np.mean(self.training_raw_rewards)
            self.writer.add_scalar('reward/training_raw_reward', avg_training_raw_reward, global_step=self.total_steps)

        self.writer.add_scalar('loss/contrastive_loss', avg_contrastive_loss, global_step=self.total_steps)


        np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed),
                np.array(self.win_rates))

    def run_episode(self, evaluate=False):
        win_tag = False
        episode_reward_raw = 0
        episode_reward_norm = 0
        avg_delay = 0
        avg_acc = 0
        avg_bi = 0
        avg_mu = 0
        self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:
            self.agent_n.actor.rnn_hidden = None
            # self.agent_n.critic.rnn_hidden = None

        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()

            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, avail_a_n, evaluate=evaluate)
            v_n = self.agent_n.get_value(s, obs_n)
            r, done, info = self.env.step(dict(zip(self.env.agent_ids, a_n)))
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False

            episode_reward_raw += r

            if not evaluate:
                r_norm = r
                if self.args.use_reward_norm:
                    r_norm = self.reward_norm(r)
                elif self.args.use_reward_scaling:
                    r_norm = self.reward_scaling(r)

                episode_reward_norm += r_norm

                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, avail_a_n, a_n, a_logprob_n, r_norm, dw)
            else:
                pass

            avg_delay += info['avg_delay']
            avg_acc += info['accuracy']
            avg_bi += info['bi']
            avg_mu += info['mu']

            if done:
                break


        avg_episode_reward_raw = episode_reward_raw / (episode_step + 1)
        avg_episode_reward_norm = 0 # Placeholder
        if not evaluate:
            r_norm_sum = 0

            temp_reward_norm = Normalization(shape=1)
            for i in range(episode_step + 1):

                r_norm_sum += temp_reward_norm(self.replay_buffer.buffer['r'][self.replay_buffer.episode_num][i][0])
            avg_episode_reward_norm = r_norm_sum / (episode_step + 1)

        avg_delay = avg_delay / (episode_step + 1)
        avg_acc = avg_acc / (episode_step + 1)
        avg_bi = avg_bi/(episode_step + 1)
        avg_mu = avg_mu/(episode_step + 1)
        if not evaluate:
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            v_n = self.agent_n.get_value(s, obs_n)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)
            return avg_episode_reward_raw, avg_episode_reward_norm, episode_step + 1, avg_delay, avg_acc, avg_bi, avg_mu
        else:
            return win_tag, avg_episode_reward_raw, episode_step + 1, avg_delay, avg_acc, avg_bi, avg_mu


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO")
    parser.add_argument("--max_train_steps", type=int, default=int(1.0e6))
    parser.add_argument("--evaluate_freq", type=float, default=500)
    parser.add_argument("--evaluate_times", type=float, default=32)
    parser.add_argument("--save_freq", type=int, default=int(1e5))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mini_batch_size", type=int, default=8)
    parser.add_argument("--rnn_hidden_dim", type=int, default=64)
    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--K_epochs", type=int, default=10)
    parser.add_argument("--use_adv_norm", type=bool, default=True)
    parser.add_argument("--use_reward_norm", type=bool, default=True)
    parser.add_argument("--use_reward_scaling", type=bool, default=False)
    parser.add_argument("--entropy_coef", type=float, default=0.005)
    parser.add_argument("--use_lr_decay", type=bool, default=True)
    parser.add_argument("--use_grad_clip", type=bool, default=True)
    parser.add_argument("--use_orthogonal_init", type=bool, default=True)
    parser.add_argument("--set_adam_eps", type=bool, default=True)
    parser.add_argument("--use_relu", type=bool, default=True)
    parser.add_argument("--use_rnn", type=bool, default=False)
    parser.add_argument("--add_agent_id", type=bool, default=False)
    parser.add_argument("--use_agent_specific", type=bool, default=False)
    parser.add_argument("--use_value_clip", type=bool, default=True)
    parser.add_argument("--critic_coef", type=float, default=0.5, help="critic loss coefficient")
    parser.add_argument("--contrastive_update_freq", type=int, default=10, help="frequency of updating contrastive learner")
    parser.add_argument("--min_entropy_coef", type=float, default=0.001, help="minimum entropy coefficient")
    parser.add_argument("--stop_contrastive_after_ratio", type=float, default=0.4, help="stop contrastive updates after this training ratio (0-1)")

    args = parser.parse_args()
    runner = Runner_MAPPO(args, env_name='security_mec', number=1, seed=43)
    runner.run()

