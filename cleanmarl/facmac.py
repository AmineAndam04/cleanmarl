import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
import tyro
import random
from env.pettingzoo_wrapper import PettingZooWrapper
from env.smaclite_wrapper import SMACliteWrapper
from env.lbf import LBFWrapper
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import datetime
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    env_type: str = "smaclite"  # "pz"
    """ Pettingzoo, SMAClite ... """
    env_name: str = "3m"  # "simple_spread_v3" #"pursuit_v4"
    """ Name of the environment """
    env_family: str = "mpe"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    gamma: float = 0.99
    """ Discount factor"""
    buffer_size: int = 5000
    """ The number of episodes in the replay buffer"""
    batch_size: int = 10
    """ Batch size"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    actor_hidden_dim: int = 32
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 64
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    hyper_dim: int = 32
    """ Hidden dimension of hyper-network"""
    train_freq: int = 1
    """ Train the network each «train_freq» step in the environment"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0008
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0008
    """ Learning rate for the critic"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    target_network_update_freq: int = 1
    """ Update the target network each target_network_update_freq» step in the environment"""
    polyak: float = 0.005
    """ Polyak coefficient when using polyak averaging for target network update"""
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    start_e: float = 0.5
    """ The starting value of epsilon. See Architecture & Training in COMA's paper Sec. 5"""
    end_e: float = 0.002
    """ The end value of epsilon. See Architecture & Training in COMA's paper Sec. 5"""
    exploration_fraction: float = 750
    """ The number of training steps it takes from to go from start_e to  end_e"""
    log_every: int = 10
    """ Logging steps """
    eval_steps: int = 50
    """ Evaluate the policy each «eval_steps» steps"""
    num_eval_ep: int = 5
    """ Number of evaluation episodes"""
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


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))

    def act(self, x, avail_action, hard=False, eps=0):
        x = self.logits(x, avail_action)
        if eps > 0:
            masked_eps = (avail_action) * (eps / avail_action.sum(dim=-1, keepdim=True))
            probs = (1 - eps) * F.gumbel_softmax(logits=x) + masked_eps
            distribution = Categorical(probs)
            actions = distribution.sample()
        else:
            actions = F.gumbel_softmax(logits=x, hard=hard)
        return actions

    def logits(self, x, avail_action=None):
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, -1e9)
        return x


class Qnetwrok(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, 1)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, s_dim, hidden_dim):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.hypernet_weight_1 = nn.Linear(s_dim, n_agents * hidden_dim)
        self.hypernet_bias_1 = nn.Linear(s_dim, hidden_dim)
        self.hypernet_weight_2 = nn.Linear(s_dim, hidden_dim)
        self.hypernet_bias_2 = nn.Sequential(
            nn.Linear(s_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, Q, s):
        Q = Q.reshape(-1, 1, self.n_agents)
        W1 = torch.abs(self.hypernet_weight_1(s))
        # W1 = self.hypernet_weight_1(s)
        W1 = W1.reshape(-1, self.n_agents, self.hidden_dim)
        b1 = self.hypernet_bias_1(s)
        b1 = b1.reshape(-1, 1, self.hidden_dim)
        Q = nn.functional.elu(torch.bmm(Q, W1) + b1)

        W2 = torch.abs(self.hypernet_weight_2(s))
        # W2 = self.hypernet_weight_2(s)
        W2 = W2.reshape(-1, self.hidden_dim, 1)
        b2 = self.hypernet_bias_2(s)
        b2 = b2.reshape(-1, 1, 1)
        Q_tot = torch.bmm(Q, W2) + b2
        return Q_tot


class ReplayBuffer:
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
        self.size = 0

    def store(self, episode):
        for key, values in episode.items():
            episode[key] = torch.from_numpy(np.stack(values)).float().to(self.device)
        self.episodes[self.pos] = (
            episode  # {"obs": [],"actions":[],"reward":[],"states":[],"done":[],"avail_actions":[]}
        )
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = [self.episodes[i] for i in indices]
        lengths = [len(episode["obs"]) for episode in batch]
        # print(lengths)
        max_length = max(lengths)
        obs = torch.zeros((batch_size, max_length, self.num_agents, self.obs_space)).to(
            self.device
        )
        avail_actions = torch.zeros(
            (batch_size, max_length, self.num_agents, self.action_space)
        ).to(self.device)
        actions = torch.zeros(
            (batch_size, max_length, self.num_agents, self.action_space)
        ).to(self.device)
        reward = torch.zeros((batch_size, max_length)).to(self.device)
        states = torch.zeros((batch_size, max_length, self.state_space)).to(self.device)
        done = torch.ones((batch_size, max_length)).to(self.device)
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool).to(self.device)

        for i in range(batch_size):
            length = lengths[i]
            obs[i, :length] = batch[i]["obs"]
            avail_actions[i, :length] = batch[i]["avail_actions"]
            actions[i, :length] = batch[i]["actions"]
            reward[i, :length] = batch[i]["reward"]
            states[i, :length] = batch[i]["states"]
            done[i, :length] = batch[i]["done"]
            mask[i, :length] = 1

        if self.normalize_reward:
            mu = torch.mean(reward[mask])
            std = torch.std(reward[mask])
            reward[mask.bool()] = (reward[mask] - mu) / (std + 1e-6)

        return (
            obs.float(),
            actions.float(),
            reward.float(),
            states.float(),
            avail_actions.bool(),
            done.float(),
            mask,
        )


def environment(env_type, env_name, env_family, agent_ids, kwargs):
    if env_type == "pz":
        env = PettingZooWrapper(
            family=env_family, env_name=env_name, agent_ids=agent_ids, **kwargs
        )
    elif env_type == "smaclite":
        env = SMACliteWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    elif env_type == "lbf":
        env = LBFWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)

    return env


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.tensor(norms), d)
    return total_norm_d


def soft_update(target_net, utility_net, polyak):
    for target_param, param in zip(target_net.parameters(), utility_net.parameters()):
        target_param.data.copy_(
            polyak * param.data + (1.0 - polyak) * target_param.data
        )


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Set the random seed
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

    actor = Actor(
        input_dim=env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=env.get_action_size(),
    ).to(device)
    target_actor = copy.deepcopy(actor).to(device)

    print(env.get_obs_size() + env.get_action_size())
    critic = Qnetwrok(
        input_dim=env.get_obs_size() + env.get_action_size(),
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
    ).to(device)
    target_critic = copy.deepcopy(critic).to(device)

    mixer = MixingNetwork(
        n_agents=env.n_agents, s_dim=env.get_state_size(), hidden_dim=args.hyper_dim
    ).to(device)
    target_mixer = copy.deepcopy(mixer).to(device)

    ## initialize the optimizer
    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(
        list(critic.parameters()) + list(mixer.parameters()),
        lr=args.learning_rate_critic,
    )

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"FACMAC-{run_name}",
        )
    writer = SummaryWriter(f"runs/FACMAC-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
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
    num_episode = 0
    num_updates = 0
    step = 0
    while step < args.total_timesteps:
        episode = {
            "obs": [],
            "actions": [],
            "reward": [],
            "states": [],
            "done": [],
            "avail_actions": [],
        }
        epsilon = linear_schedule(
            args.start_e, args.end_e, args.exploration_fraction, num_updates
        )
        obs, _ = env.reset()
        ep_reward, ep_length = 0, 0
        done, truncated = False, False
        while not done and not truncated:
            avail_action = env.get_avail_actions()
            state = env.get_state()
            with torch.no_grad():
                actions = actor.act(
                    torch.from_numpy(obs).float().to(device),
                    eps=epsilon,
                    avail_action=torch.from_numpy(avail_action).bool().to(device),
                    hard=True,
                ).cpu()  ## These are one hot-vectors
                if epsilon > 0:
                    actions_to_take = actions.clone()
                    actions = F.one_hot(
                        actions.long(), num_classes=env.get_action_size()
                    )
                else:
                    actions_to_take = torch.argmax(actions, dim=-1)
            next_obs, reward, done, truncated, infos = env.step(actions_to_take)

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
        rb.store(episode)

        num_episode += 1
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)
        if args.env_type == "smaclite":
            ep_stats.append(infos)  ## Add battle won for smaclite

        if num_episode % args.log_every == 0:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            writer.add_scalar("rollout/num_episodes", num_episode, step)
            writer.add_scalar("rollout/epsilon", epsilon, step)
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "rollout/battle_won",
                    np.mean(np.mean([info["battle_won"] for info in ep_stats])),
                    step,
                )
            ep_rewards = []
            ep_lengths = []
            ep_stats = []
        if num_episode > args.batch_size:
            if num_episode % args.train_freq == 0:
                (
                    batch_obs,
                    batch_action,
                    batch_reward,
                    batch_states,
                    batch_avail_action,
                    batch_done,
                    batch_mask,
                ) = rb.sample(args.batch_size)
                ## train the critic
                critic_loss = 0
                for t in range(batch_obs.size(1)):
                    with torch.no_grad():
                        if t == (batch_obs.size(1) - 1):
                            targets = batch_reward[:, t]
                        else:
                            actions_from_target_actor = target_actor.act(
                                batch_obs[:, t + 1],
                                avail_action=batch_avail_action[:, t + 1],
                                hard=True,
                            )
                            qvals_from_taget_utility = target_critic(
                                torch.cat(
                                    (batch_obs[:, t + 1], actions_from_target_actor),
                                    dim=-1,
                                )
                            ).squeeze()
                            q_tot_from_target_mixer = target_mixer(
                                Q=qvals_from_taget_utility, s=batch_states[:, t + 1]
                            ).squeeze()
                            q_tot_from_target_mixer = torch.nan_to_num(
                                q_tot_from_target_mixer, nan=0.0
                            )
                            targets = (
                                batch_reward[:, t]
                                + args.gamma
                                * (1 - batch_done[:, t])
                                * q_tot_from_target_mixer
                            )
                    q_values = critic(
                        torch.cat((batch_obs[:, t], batch_action[:, t]), dim=-1)
                    ).squeeze()
                    q_tot = mixer(Q=q_values, s=batch_states[:, t]).squeeze()
                    critic_loss += F.mse_loss(
                        targets[batch_mask[:, t]], q_tot[batch_mask[:, t]]
                    ) * (batch_mask[:, t].sum())
                critic_loss /= batch_mask.sum()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_gradients = norm_d([p.grad for p in critic.parameters()], 2)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(
                        critic.parameters(), max_norm=args.clip_gradients
                    )
                    torch.nn.utils.clip_grad_norm_(
                        mixer.parameters(), max_norm=args.clip_gradients
                    )
                critic_optimizer.step()

                perm = torch.randperm(batch_obs.size(1) - 1)
                actor_loss = 0
                for t in perm:
                    actions = actor.act(
                        batch_obs[:, t],
                        avail_action=batch_avail_action[:, t],
                        hard=False,
                    )
                    q_values = critic(
                        torch.cat((batch_obs[:, t], actions), dim=-1)
                    ).squeeze()
                    q_tot = mixer(Q=q_values, s=batch_states[:, t]).squeeze()
                    actor_loss -= q_tot[batch_mask[:, t]].sum()
                actor_loss /= batch_mask.sum()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_gradients = norm_d([p.grad for p in actor.parameters()], 2)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), max_norm=args.clip_gradients
                    )
                actor_optimizer.step()
                num_updates += 1

                writer.add_scalar("train/critic_loss", critic_loss, step)
                writer.add_scalar("train/actor_loss", actor_loss, step)
                writer.add_scalar("train/actor_gradients", actor_gradients, step)
                writer.add_scalar("train/critic_gradients", critic_gradients, step)
                writer.add_scalar("train/num_updates", num_updates, step)
            if num_episode % args.target_network_update_freq == 0:
                soft_update(
                    target_net=target_actor, utility_net=actor, polyak=args.polyak
                )
                soft_update(
                    target_net=target_critic, utility_net=critic, polyak=args.polyak
                )
                soft_update(
                    target_net=target_mixer, utility_net=mixer, polyak=args.polyak
                )
            if num_episode % args.eval_steps == 0:
                eval_obs, _ = eval_env.reset()
                eval_ep = 0
                eval_ep_reward = []
                eval_ep_length = []
                eval_ep_stats = []
                current_reward = 0
                current_ep_length = 0
                while eval_ep < args.num_eval_ep:
                    with torch.no_grad():
                        logits = actor.logits(
                            torch.from_numpy(eval_obs).float().to(device),
                            avail_action=torch.tensor(eval_env.get_avail_actions())
                            .bool()
                            .to(device),
                        )
                        eval_actions = torch.argmax(logits, dim=-1)
                    next_obs_, reward, done, truncated, infos = eval_env.step(
                        eval_actions
                    )
                    current_reward += reward
                    current_ep_length += 1
                    eval_obs = next_obs_
                    if done or truncated:
                        eval_obs, _ = eval_env.reset()
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
                        np.mean(
                            np.mean([info["battle_won"] for info in eval_ep_stats])
                        ),
                        step,
                    )

    writer.close()
    if args.use_wnb:
        wandb.finish()
    env.close()
    eval_env.close()
