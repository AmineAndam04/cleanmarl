import copy
import random
import tyro
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import torch.nn.functional as F
from env.pettingzoo_wrapper import PettingZooWrapper
from env.smaclite_wrapper import SMACliteWrapper
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    env_type: str = "smaclite"
    """ Pettingzoo, SMAClite ... """
    env_name: str = "3m"
    """ Name of the environment"""
    env_family: str = "mpe"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    batch_size: int = 5
    """ Number of episodes to collect in each rollout"""
    actor_hidden_dim: int = 32
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 128
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0005
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0005
    """ Learning rate for the critic"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    td_lambda: float = 0.8
    """ TD(λ) discount factor"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    normalize_advantage: bool = True
    """ Normalize the advantage if True"""
    normalize_return: bool = False
    """ Normalize the returns if True"""
    target_network_update_freq: int = 1
    """ Update the target network each target_network_update_freq» step in the environment"""
    polyak: float = 0.005
    """ Polyak coefficient when using polyak averaging for target network update"""
    eval_steps: int = 10
    """ Evaluate the policy each «eval_steps» training steps"""
    use_tdlamda: bool = True
    """ Use TD(λ) as a target for the critic, if False use n-step returns (n=nsteps) """
    nsteps: int = 1
    """ number of stpes when using n-step returns as a target for the critic"""
    start_e: float = 0.5
    """ The starting value of epsilon. See Architecture & Training in COMA's paper Sec. 5"""
    end_e: float = 0.002
    """ The end value of epsilon. See Architecture & Training in COMA's paper Sec. 5"""
    exploration_fraction: float = 750
    """ The number of training steps it takes from to go from start_e to  end_e"""
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    tbptt: int = 10
    """Chunck size for Truncated Backpropagation Through Time tbptt"""
    log_every: int = 10
    """ Log rollout stats every log_every episode"""
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""
    entropy_coef: float = 0.001
    """ Entropy coefficient """
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""
    device: str = "cpu"
    """ Device (cpu, cuda, mps)"""
    seed: int = 1
    """ Random seed"""


class RolloutBuffer:
    def __init__(
        self,
        buffer_size,
        num_agents,
        obs_space,
        state_space,
        action_space,
        normalize_reward=False,
        device="cpu",
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.normalize_reward = normalize_reward
        self.device = device
        self.episodes = [None] * buffer_size
        self.pos = 0

    def add(self, episode):
        for key, values in episode.items():
            episode[key] = torch.from_numpy(np.stack(values)).float().to(self.device)
        self.episodes[self.pos] = episode
        self.pos += 1

    def get_batch(self):
        self.pos = 0
        lengths = [len(episode["obs"]) for episode in self.episodes]
        max_length = max(lengths)
        obs = torch.zeros(
            (self.buffer_size, max_length, self.num_agents, self.obs_space)
        ).to(self.device)
        avail_actions = torch.zeros(
            (self.buffer_size, max_length, self.num_agents, self.action_space)
        ).to(self.device)
        actions = torch.zeros((self.buffer_size, max_length, self.num_agents)).to(
            self.device
        )
        reward = torch.zeros((self.buffer_size, max_length)).to(self.device)
        states = torch.zeros((self.buffer_size, max_length, self.state_space)).to(
            self.device
        )
        done = torch.zeros((self.buffer_size, max_length)).to(self.device)
        mask = torch.zeros(self.buffer_size, max_length, dtype=torch.bool).to(
            self.device
        )
        for i in range(self.buffer_size):
            length = lengths[i]
            obs[i, :length] = self.episodes[i]["obs"]
            avail_actions[i, :length] = self.episodes[i]["avail_actions"]
            actions[i, :length] = self.episodes[i]["actions"]
            reward[i, :length] = self.episodes[i]["reward"]
            states[i, :length] = self.episodes[i]["states"]
            done[i, :length] = self.episodes[i]["done"]
            mask[i, :length] = 1
        if self.normalize_reward:
            mu = torch.mean(reward[mask])
            std = torch.std(reward[mask])
            reward[mask.bool()] = (reward[mask] - mu) / (std + 1e-6)
        self.episodes = [None] * self.buffer_size
        return (
            obs.float(),
            actions.long(),
            reward.float(),
            states.float(),
            avail_actions.bool(),
            done.float(),
            mask,
        )


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def act(self, x, h=None, eps=0, avail_action=None):
        x = self.fc1(x)
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        h = self.gru(x, h)
        x = self.fc2(h)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, float("-inf"))
        masked_eps = (avail_action) * (eps / avail_action.sum(dim=-1, keepdim=True))
        probs = (1 - eps) * F.softmax(x, dim=-1) + masked_eps
        distribution = Categorical(probs)
        action = distribution.sample()
        return action, h

    def logits(self, x, h=None, eps=0, avail_action=None):
        x = self.fc1(x)
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        h = self.gru(x, h)
        x = self.fc2(h)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, float("-inf"))
        masked_eps = (avail_action) * (eps / avail_action.sum(dim=-1, keepdim=True))
        probs = (1 - eps) * F.softmax(x, dim=-1) + masked_eps
        return probs, h


class Critic(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_layer, output_dim, num_agents
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))

    def forward(self, state, observations, actions, avail_actions=None):
        if state.dim() < 2:
            state = state.unsqueeze(0)
            observations = observations.unsqueeze(0)
            actions = actions.unsqueeze(0)
            if avail_actions is not None:
                avail_actions = avail_actions.unsqueeze(0)
        x = self.coma_inputs(state, observations, actions)
        for layer in self.layers:
            x = layer(x)
        if avail_actions is not None:
            x = x.masked_fill(~avail_actions, float("-inf"))
        return x.squeeze()

    def coma_inputs(self, state, observations, actions):
        coma_inputs = torch.zeros((state.size(0), self.num_agents, self.input_dim)).to(
            state.device
        )
        coma_inputs[:, :, : state.size(-1)] = state.unsqueeze(1)
        coma_inputs[:, :, state.size(-1) : state.size(-1) + observations.size(-1)] = (
            observations
        )
        one_hot = F.one_hot(actions.long(), num_classes=self.output_dim).float()
        mask = ~torch.eye(self.num_agents, dtype=torch.bool)
        oh = one_hot.unsqueeze(1).expand(
            state.size(0), self.num_agents, self.num_agents, self.output_dim
        )
        oh = oh[mask.unsqueeze(0).expand(state.size(0), -1, -1)]
        oh = oh.view(
            state.size(0), self.num_agents, (self.num_agents - 1) * self.output_dim
        )
        coma_inputs[:, :, state.size(-1) + observations.size(-1) :] = oh
        return coma_inputs


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def environment(env_type, env_name, env_family, agent_ids, kwargs):
    if env_type == "pz":
        env = PettingZooWrapper(
            family=env_family, env_name=env_name, agent_ids=agent_ids, **kwargs
        )
    elif env_type == "smaclite":
        env = SMACliteWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)

    return env


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.tensor(norms), d)
    return total_norm_d


def soft_update(target_net, critic_net, polyak):
    for target_param, param in zip(target_net.parameters(), critic_net.parameters()):
        target_param.data.copy_(
            polyak * param.data + (1.0 - polyak) * target_param.data
        )


def get_coma_critic_input_dim(env):
    critic_input_dim = (
        env.get_obs_size()
        + env.get_state_size()
        + (env.n_agents - 1) * env.get_action_size()
    )
    return critic_input_dim


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Set the seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    ## import the environment
    kwargs = {}  # {"render_mode":'human',"shared_reward":False}
    env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=kwargs,
    )
    eval_env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=kwargs,
    )

    # Initialize the actor, critic and target-critic networks
    actor = Actor(
        input_dim=env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        output_dim=env.get_action_size(),
    ).to(device)
    critic_input_dim = get_coma_critic_input_dim(env)
    critic = Critic(
        input_dim=critic_input_dim,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        output_dim=env.get_action_size(),
        num_agents=env.n_agents,
    ).to(device)
    target_critic = copy.deepcopy(critic).to(device)

    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"COMA-lstm-{run_name}",
        )
    writer = SummaryWriter(f"runs/COMA-lstm-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    rb = RolloutBuffer(
        buffer_size=args.batch_size,
        obs_space=env.get_obs_size(),
        state_space=env.get_state_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        normalize_reward=args.normalize_reward,
        device=device,
    )
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    step = 0
    training_step = 0
    while step < args.total_timesteps:
        num_episode = 0
        while num_episode < args.batch_size:
            epsilon = linear_schedule(
                args.start_e, args.end_e, args.exploration_fraction, training_step
            )
            episode = {
                "obs": [],
                "actions": [],
                "reward": [],
                "states": [],
                "done": [],
                "avail_actions": [],
            }
            obs, _ = env.reset()
            ep_reward, ep_length = 0, 0
            done, truncated = False, False
            h = None
            while not done and not truncated:
                avail_action = env.get_avail_actions()
                state = torch.from_numpy(env.get_state()).float()
                with torch.no_grad():
                    actions, h = actor.act(
                        torch.from_numpy(obs).float().to(device),
                        h=h,
                        eps=epsilon,
                        avail_action=torch.from_numpy(avail_action).bool().to(device),
                    )
                    actions = actions.cpu()
                next_obs, reward, done, truncated, infos = env.step(actions)
                ep_reward += reward
                ep_length += 1
                step += 1
                episode["obs"].append(obs)
                episode["actions"].append(actions)
                episode["reward"].append(reward)
                episode["done"].append(done)
                episode["avail_actions"].append(avail_action)
                episode["states"].append(state)

                obs = next_obs

            rb.add(episode)
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            if args.env_type == "smaclite":
                ep_stats.append(infos)  ## Add battle won for smaclite
            num_episode += 1

        ## logging
        if len(ep_rewards) > args.log_every:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            writer.add_scalar("rollout/epsilon", epsilon, step)
            writer.add_scalar(
                "rollout/num_episodes", (training_step + 1) * args.batch_size, step
            )
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "rollout/battle_won",
                    np.mean([info["battle_won"] for info in ep_stats]),
                    step,
                )
            ep_rewards = []
            ep_lengths = []
            ep_stats = []

        ## Collate episodes in buffer into single batch
        b_obs, b_actions, b_reward, b_states, b_avail_actions, b_done, b_mask = (
            rb.get_batch()
        )
        ### 1. Compute TD(λ) from "Reconciling λ-Returns with Experience Replay"(https://arxiv.org/pdf/1810.09967 Equation 3)
        with torch.no_grad():
            return_lambda = torch.zeros_like(b_actions).float()
            if args.use_tdlamda:
                for ep_idx in range(return_lambda.size(0)):
                    ep_len = b_mask[ep_idx].sum()
                    last_return_lambda = 0
                    for t in reversed(range(ep_len)):
                        if t == (ep_len - 1):
                            next_action_value = 0
                        else:
                            next_action_value = target_critic(
                                state=b_states[ep_idx, t + 1],
                                observations=b_obs[ep_idx, t + 1],
                                actions=b_actions[ep_idx, t + 1],
                                avail_actions=b_avail_actions[ep_idx, t + 1],
                            )
                            next_action_value = torch.gather(
                                next_action_value,
                                dim=-1,
                                index=b_actions[ep_idx, t + 1].unsqueeze(-1),
                            ).squeeze()
                            # next_action_value, _ = next_action_value.max(dim=-1)

                        return_lambda[ep_idx, t] = last_return_lambda = b_reward[
                            ep_idx, t
                        ] + args.gamma * (
                            args.td_lambda * last_return_lambda
                            + (1 - args.td_lambda) * next_action_value
                        )
            else:
                for ep_idx in range(return_lambda.size(0)):
                    ep_len = b_mask[ep_idx].sum()
                    for t in range(ep_len):
                        if t < (ep_len - args.nsteps):
                            return_t_n = b_reward[ep_idx, t : t + args.nsteps]
                            discounts = torch.tensor(
                                [args.gamma**i for i in range(return_t_n.size(-1))]
                            )
                            return_t_n = (return_t_n * discounts).sum(-1)
                            action_value_t_n = target_critic(
                                state=b_states[ep_idx, t + args.nsteps],
                                observations=b_obs[ep_idx, t + args.nsteps],
                                actions=b_actions[ep_idx, t + args.nsteps],
                                avail_actions=b_avail_actions[ep_idx, t + args.nsteps],
                            )
                            action_value_t_n = torch.gather(
                                action_value_t_n,
                                dim=-1,
                                index=b_actions[ep_idx, t + args.nsteps].unsqueeze(-1),
                            ).squeeze()
                            return_t_n = (
                                return_t_n + args.gamma**args.nsteps * action_value_t_n
                            )

                        else:
                            return_t_n = b_reward[ep_idx, t:]
                            discounts = torch.tensor(
                                [args.gamma**i for i in range(return_t_n.size(-1))]
                            )
                            return_t_n = (return_t_n * discounts).sum(-1)
                            return_t_n = return_t_n.expand(eval_env.n_agents)
                        return_lambda[ep_idx, t] = return_t_n

        if args.normalize_return:
            ret_mu = return_lambda.mean(dim=-1)[b_mask].mean()
            ret_std = return_lambda.mean(dim=-1)[b_mask].std()
            return_lambda = (return_lambda - ret_mu) / ret_std
        ### 2. Update the critic
        cr_loss = 0
        for t in range(b_obs.size(1)):
            b_q_values = critic(
                state=b_states[:, t], observations=b_obs[:, t], actions=b_actions[:, t]
            )
            b_q_values = torch.gather(
                b_q_values, dim=-1, index=b_actions[:, t].unsqueeze(-1)
            ).squeeze()
            q_targets = return_lambda[:, t]
            critic_loss = F.mse_loss(b_q_values[b_mask[:, t]], q_targets[b_mask[:, t]])
            cr_loss += critic_loss * b_mask[:, t].sum()

        critic_optimizer.zero_grad()
        cr_loss = cr_loss / b_mask.sum()
        cr_loss.backward()
        critic_gradients = norm_d([p.grad for p in critic.parameters()], 2)
        if args.clip_gradients > 0:
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(), max_norm=args.clip_gradients
            )
        critic_optimizer.step()

        training_step += 1
        if training_step % args.target_network_update_freq == 0:
            soft_update(target_net=target_critic, critic_net=critic, polyak=args.polyak)
        ### 3. Update actor
        actor_losses = 0
        entropies = 0
        actor_gradients = []
        h = None
        truncated_actor_loss = None
        actor_loss_denominator = None
        T = None
        for t in range(b_obs.size(1)):
            b_obs_t = b_obs[:, t].reshape(args.batch_size * eval_env.n_agents, -1)
            b_avail_actions_t = b_avail_actions[:, t].reshape(
                args.batch_size * eval_env.n_agents, -1
            )
            pi, h = actor.logits(b_obs_t, h=h, avail_action=b_avail_actions_t)
            pi = pi.reshape(args.batch_size, eval_env.n_agents, -1)
            log_pi = torch.log(pi + 1e-8)
            entropy_loss = -(pi * log_pi).mean(dim=-1)[b_mask[:, t]]
            entropy_loss = entropy_loss.sum()
            entropies += entropy_loss
            q_values = critic(
                state=b_states[:, t], observations=b_obs[:, t], actions=b_actions[:, t]
            )
            q_values = q_values.detach()
            coma_baseline = (pi * q_values).sum(dim=-1)
            current_q = torch.gather(
                q_values, dim=-1, index=b_actions[:, t].unsqueeze(-1)
            ).squeeze()
            advantage = (current_q - coma_baseline).detach()
            if args.normalize_advantage and b_actions[:, t].sum() > eval_env.n_agents:
                advantage = (advantage - advantage[b_mask[:, t]].mean()) / (
                    advantage[b_mask[:, t]].std() + 1e-8
                )
            log_pi = torch.gather(
                log_pi, dim=-1, index=b_actions[:, t].unsqueeze(-1)
            ).squeeze()
            actor_loss = -(log_pi[b_mask[:, t]] * advantage[b_mask[:, t]]).sum()
            actor_loss = actor_loss - args.entropy_coef * entropy_loss
            actor_losses += actor_loss
            if truncated_actor_loss is None:
                truncated_actor_loss = actor_loss
                actor_loss_denominator = b_mask[:, t].sum()
                T = 1
            else:
                truncated_actor_loss += actor_loss
                actor_loss_denominator += b_mask[:, t].sum()
                T += 1
            if ((t + 1) % args.tbptt == 0) or (t == (b_obs.size(1) - 1)):
                # For more details: https://d2l.ai/chapter_recurrent-neural-networks/bptt.html#equation-eq-bptt-partial-ht-wh-gen
                truncated_actor_loss = truncated_actor_loss / (
                    actor_loss_denominator * T
                )
                actor_optimizer.zero_grad()
                truncated_actor_loss.backward()
                tbptt_actor_gradients = norm_d([p.grad for p in actor.parameters()], 2)
                actor_gradients.append(tbptt_actor_gradients)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), max_norm=args.clip_gradients
                    )
                actor_optimizer.step()
                truncated_actor_loss = None
                h = h.detach()

        actor_losses /= b_mask.sum()
        entropies /= b_mask.sum()

        writer.add_scalar("train/critic_loss", cr_loss, step)
        writer.add_scalar("train/actor_loss", actor_losses, step)
        writer.add_scalar("train/entropy", entropies, step)
        writer.add_scalar("train/critic_gradients", critic_gradients, step)
        writer.add_scalar("train/actor_gradients", np.mean(actor_gradients), step)
        writer.add_scalar("train/epsilon", epsilon, step)
        writer.add_scalar("train/num_updates", training_step, step)

        if training_step % args.eval_steps == 0:
            eval_obs, _ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            h_eval = None
            while eval_ep < args.num_eval_ep:
                with torch.no_grad():
                    actions, h_eval = actor.act(
                        torch.from_numpy(eval_obs).float().to(device),
                        h_eval,
                        avail_action=torch.tensor(
                            eval_env.get_avail_actions(), dtype=torch.bool
                        ).to(device),
                    )
                next_obs_, reward, done, truncated, infos = eval_env.step(actions.cpu())
                current_reward += reward
                current_ep_length += 1
                eval_obs = next_obs_
                if done or truncated:
                    eval_obs, _ = eval_env.reset()
                    h_eval = None
                    eval_ep_reward.append(current_reward)
                    eval_ep_length.append(current_ep_length)
                    eval_ep_stats.append(infos)
                    current_reward = 0
                    current_ep_length = 0
                    eval_ep += 1
            writer.add_scalar("eval/ep_reward", np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/std_ep_reward", np.std(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length", np.mean(eval_ep_length), step)
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "eval/battle_won",
                    np.mean([info["battle_won"] for info in eval_ep_stats]),
                    step,
                )

    writer.close()
    if args.use_wnb:
        wandb.finish()
    env.close()
    eval_env.close()
