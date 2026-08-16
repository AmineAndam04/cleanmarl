import copy
import datetime
import json
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from marl_envs.vec_envs import SyncVectorEnv
from marl_envs.wrappers import AddAgentIDVec, NormalizeVecObservation, RecordEpisodeStatistics
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    # Environment
    env_type: str = "smaclite"
    """ pz(for Pettingzoo), smaclite, lbf, rware, smac, smacv2 """
    env_name: str = "3m"
    """ Name of the environment """
    env_family: str = "mpe"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    normalize_obs: bool = False
    """ NNormalize the observations if True"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    max_episode_steps: int = 150
    "Maximum steps per episode"
    # Network
    hidden_dim: int = 32
    """ Hidden dimension"""
    hyper_dim: int = 32
    """ Hidden dimension of hyper-network"""
    # Training
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    train_freq: int = 1
    """ Train the network each «train_freq» step in the environment"""
    buffer_size: int = 5000
    """ The number of episodes in the replay buffer"""
    batch_size: int = 3
    """ Number of sampled episodes"""
    tbptt: int = 10
    """Chunck size for Truncated Backpropagation Through Time"""
    gamma: float = 0.99
    """ Discount factor"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate: float = 0.0008
    """ Learning rate"""
    target_network_update_freq: int = 1
    """ Update the target network each target_network_update_freq» step in the environment"""
    polyak: float = 0.005
    """ Polyak coefficient when using polyak averaging for target network update"""
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    start_e: float = 1
    """ The starting value of epsilon, for exploration"""
    end_e: float = 0.05
    """ The end value of epsilon, for exploration"""
    exploration_fraction: float = 0.05
    """ The fraction of «total-timesteps» it takes from to go from start_e to  end_e"""
    device: str = "cpu"
    """ Device (cpu, cuda, mps)"""
    seed: int = 1
    """ Random seed"""
    # Logging
    work_dir: str = "runs"
    """ Folder to save logs, weights ..."""
    save_model: bool = False
    """ If True, save the weights of the agents and hyperparameters"""
    exp_name: str = "v1"
    """ Used for logging"""
    log_every: int = 10
    """ Log rollout stats every <log_every> episode """
    eval_steps: int = 50
    """ Evaluate the policy each «eval_steps» episode"""
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""


class Qnetwrok(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc2 = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x, h=None, avail_action=None):
        x = self.fc1(x)
        if h is None:
            h = (
                torch.zeros(1, x.size(0), self.hidden_dim, device=x.device),
                torch.zeros(1, x.size(0), self.hidden_dim, device=x.device),
            )
        if x.dim() < 3:
            x = x.unsqueeze(1)
            if avail_action is not None:
                avail_action = avail_action.unsqueeze(1)
        x, h = self.lstm(x, h)
        x = self.fc2(x)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, -1e8)
        return x, h


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
        W1 = W1.reshape(-1, self.n_agents, self.hidden_dim)
        b1 = self.hypernet_bias_1(s)
        b1 = b1.reshape(-1, 1, self.hidden_dim)
        Q = nn.functional.elu(torch.bmm(Q, W1) + b1)

        W2 = torch.abs(self.hypernet_weight_2(s))
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
        device="cpu",
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.device = device
        self.episodes = [None] * buffer_size
        self.pos = 0
        self.size = 0

    def store(self, episode):
        for key, values in episode.items():
            episode[key] = torch.from_numpy(np.stack(values))
        self.episodes[self.pos] = episode
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        batch = [self.episodes[i] for i in indices]
        lengths = [len(episode["obs"]) - 1 for episode in batch]
        max_length = max(lengths)
        obs = torch.zeros((batch_size, max_length, self.num_agents, self.obs_space)).float().to(self.device)
        states = torch.zeros((batch_size, max_length, self.state_space)).float().to(self.device)
        actions = torch.zeros((batch_size, max_length, self.num_agents)).int().to(self.device)
        reward = torch.zeros((batch_size, max_length)).float().to(self.device)
        next_obs = (
            torch.zeros((batch_size, max_length, self.num_agents, self.obs_space)).float().to(self.device)
        )
        next_states = torch.zeros((batch_size, max_length, self.state_space)).float().to(self.device)
        next_avail_actions = (
            torch.zeros((batch_size, max_length, self.num_agents, self.action_space)).bool().to(self.device)
        )
        done = torch.ones((batch_size, max_length)).int().to(self.device)
        mask = torch.zeros(batch_size, max_length).bool().to(self.device)
        for i in range(batch_size):
            length = lengths[i]
            obs[i, :length] = batch[i]["obs"][:-1]
            states[i, :length] = batch[i]["states"][:-1]
            actions[i, :length] = batch[i]["actions"]
            reward[i, :length] = batch[i]["reward"]
            next_obs[i, :length] = batch[i]["obs"][1:]
            next_states[i, :length] = batch[i]["states"][1:]
            next_avail_actions[i, :length] = batch[i]["avail_actions"][1:]
            done[i, :length] = batch[i]["done"]
            mask[i, :length] = 1
        return (
            obs.permute(0, 2, 1, 3),
            actions.permute(0, 2, 1),
            reward,
            next_obs.permute(0, 2, 1, 3),
            states,
            next_states,
            next_avail_actions.permute(0, 2, 1, 3),
            done,
            mask,
        )


def make_env(args, kwargs, eval=False):
    def env_fn():
        if args.env_type == "pz":
            from marl_envs import PettingZooInterface  # noqa: PLC0415

            env = PettingZooInterface(
                family=args.env_family,
                max_episode_steps=args.max_episode_steps,
                env_name=args.env_name,
                **kwargs,
            )
        elif args.env_type == "smaclite":
            from marl_envs import SMACliteInterface  # noqa: PLC0415

            env = SMACliteInterface(
                env_name=args.env_name, max_episode_steps=args.max_episode_steps, **kwargs
            )
        elif args.env_type == "lbf":
            from marl_envs import LBFInterface  # noqa: PLC0415

            env = LBFInterface(env_name=args.env_name, max_episode_steps=args.max_episode_steps, **kwargs)
        elif args.env_type == "rware":
            from marl_envs import RWAREInterface  # noqa: PLC0415

            env = RWAREInterface(env_name=args.env_name, max_episode_steps=args.max_episode_steps, **kwargs)
        elif args.env_type == "smac":
            from marl_envs.wrappers import TimeLimit  # noqa: I001, PLC0415
            from marl_envs import SMACInterface  # noqa: PLC0415

            env = SMACInterface(env_name=args.env_name, seed=args.seed, **kwargs)
            env = TimeLimit(
                env=env,
                max_episode_steps=args.max_episode_steps,
            )
        elif args.env_type == "smacv2":
            from marl_envs.wrappers import TimeLimit  # noqa: I001, PLC0415
            from marl_envs import SMACv2Interface  # noqa: PLC0415

            env = SMACv2Interface(env_name=args.env_name, seed=args.seed, **kwargs)
            env = TimeLimit(
                env=env,
                max_episode_steps=args.max_episode_steps,
            )
        else:
            raise ValueError(f"{args.env_type} nor supported for VDN")

        env = RecordEpisodeStatistics(env)
        if not eval:
            if args.normalize_obs:
                from marl_envs.wrappers import NormalizeObservation  # noqa: PLC0415

                env = NormalizeObservation(env)
            if args.normalize_reward:
                from marl_envs.wrappers import NormalizeReward  # noqa: PLC0415

                env = NormalizeReward(env, gamma=args.gamma)
            if args.agent_ids:
                from marl_envs.wrappers import AddAgentID  # noqa: PLC0415

                env = AddAgentID(env)
        return env

    return env_fn


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.stack(norms), d)
    return total_norm_d


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def soft_update(target_net, utility_net, polyak):
    for target_param, param in zip(target_net.parameters(), utility_net.parameters()):
        target_param.data.copy_(polyak * param.data + (1.0 - polyak) * target_param.data)


def rms_state_dict(rms):
    return {"mean": torch.as_tensor(rms.mean).cpu().clone(), "var": torch.as_tensor(rms.var).cpu().clone()}


if __name__ == "__main__":
    # ---- Prepare for training: seed, networks, optim ... -------
    args = tyro.cli(Args)
    # Set the seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Set device
    device = torch.device(args.device)
    # Import the environment
    env = make_env(args, kwargs={})()
    eval_env = SyncVectorEnv(
        env_fns=[make_env(args, kwargs={}, eval=True) for _ in range(args.num_eval_ep)], auto_reset=False
    )
    if args.normalize_obs:  # Sync normalization statistics
        eval_env = NormalizeVecObservation(eval_env)
        eval_env.set_wrapper_attr("update_running_mean", False)
        eval_env.set_wrapper_attr("obs_rms", env.get_wrapper_attr("obs_rms"))
    if args.agent_ids:
        eval_env = AddAgentIDVec(eval_env)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 100)
    # Initialize the networks
    utility_network = Qnetwrok(
        input_dim=env.get_obs_size(),
        hidden_dim=args.hidden_dim,
        output_dim=env.get_action_size(),
    ).to(device)
    target_network = copy.deepcopy(utility_network).to(device)
    mixer = MixingNetwork(n_agents=env.n_agents, s_dim=env.get_state_size(), hidden_dim=args.hyper_dim).to(
        device
    )
    target_mixer = copy.deepcopy(mixer).to(device)
    # Initialize the optimizer
    optimizer = getattr(optim, args.optimizer)
    optimizer = optimizer(
        list(utility_network.parameters()) + list(mixer.parameters()),
        lr=args.learning_rate,
    )
    # Initialize the replay buffer
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space=env.get_obs_size(),
        state_space=env.get_state_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        device=device,
    )
    # Logging
    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{args.exp_name}__{time_token}"
    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"QMIX-lstm-{run_name}",
        )
    log_dir = f"{args.work_dir}/QMIX-lstm-{run_name}"
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )
    ep_rewards, ep_lengths, ep_stats = [], [], []
    losses, gradients = [], []
    num_episodes = 0
    step = 0
    while step < args.total_timesteps:
        episode = {"obs": [], "actions": [], "reward": [], "states": [], "done": [], "avail_actions": []}
        obs, _ = env.reset()
        avail_action = env.get_avail_actions()
        state = env.get_state()
        done, truncated = False, False
        h = None
        while not done and not truncated:
            epsilon = linear_schedule(
                args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, step
            )
            with torch.no_grad():
                q_values, h = utility_network(
                    torch.from_numpy(obs).float().to(device),
                    h=h,
                    avail_action=torch.from_numpy(avail_action).bool().to(device),
                )
                q_values = q_values.squeeze(1)
            actions = q_values.argmax(dim=-1).cpu().numpy()
            explore = np.random.random(actions.shape) < epsilon
            if explore.any():
                actions = np.where(explore, env.sample(), actions)
            # Step the environment
            next_obs, reward, done, truncated, infos = env.step(actions)
            step += 1
            episode["obs"].append(obs)
            episode["actions"].append(actions)
            episode["reward"].append(reward)
            episode["done"].append(done)
            episode["avail_actions"].append(avail_action)
            episode["states"].append(state)
            obs = next_obs
            state = env.get_state()
            avail_action = env.get_avail_actions()
        # Store last step
        episode["obs"].append(obs)
        episode["states"].append(state)
        episode["avail_actions"].append(avail_action)
        rb.store(episode)
        num_episodes += 1
        ep_rewards.append(infos["episode_stats"]["r"])
        ep_lengths.append(infos["episode_stats"]["l"])
        if "smac" in args.env_type:
            ep_stats.append(infos["battle_won"])
        # ---- Training loop -------
        if num_episodes > args.batch_size:
            if num_episodes % args.train_freq == 0:
                (
                    b_obs,
                    b_action,
                    b_reward,
                    b_next_obs,
                    b_states,
                    b_next_states,
                    b_next_avail_action,
                    b_done,
                    b_mask,
                ) = rb.sample(args.batch_size)
                ## Initialize hidden states
                h_utility = None
                with torch.no_grad():
                    _, h_target = target_network(b_obs[:, :, :1].flatten(0, 1))
                for start in range(0, b_obs.size(2), args.tbptt):
                    end = start + args.tbptt
                    with torch.no_grad():
                        q_next, h_target = target_network(
                            b_next_obs[:, :, start:end].flatten(0, 1),
                            h=h_target,
                            avail_action=b_next_avail_action[:, :, start:end].flatten(0, 1),
                        )
                        q_next_max, _ = q_next.max(dim=-1)
                        q_next_max = q_next_max.reshape(args.batch_size, env.n_agents, -1).transpose(1, 2)
                        q_tot_target = target_mixer(Q=q_next_max, s=b_next_states[:, start:end])
                        q_tot_target = q_tot_target.reshape(args.batch_size, -1)
                        targets = (
                            b_reward[:, start:end] + args.gamma * (1 - b_done[:, start:end]) * q_tot_target
                        )

                    q_values, h_utility = utility_network(b_obs[:, :, start:end].flatten(0, 1), h=h_utility)
                    q_values = torch.gather(
                        q_values, dim=-1, index=b_action[:, :, start:end].flatten(0, 1).unsqueeze(-1)
                    )
                    q_values = q_values.reshape(args.batch_size, env.n_agents, -1).transpose(1, 2)
                    q_tot = mixer(Q=q_values, s=b_states[:, start:end])
                    q_tot = q_tot.reshape(args.batch_size, -1)
                    loss = F.mse_loss(targets[b_mask[:, start:end]], q_tot[b_mask[:, start:end]])
                    optimizer.zero_grad()
                    loss.backward()
                    loss_gradients = norm_d(
                        [p.grad for p in list(utility_network.parameters()) + list(mixer.parameters())], 2
                    )
                    if args.clip_gradients > 0:
                        torch.nn.utils.clip_grad_norm_(
                            list(utility_network.parameters()) + list(mixer.parameters()), args.clip_gradients
                        )
                    optimizer.step()
                    h_utility = (h_utility[0].detach(), h_utility[1].detach())
                    losses.append(loss.item())
                    gradients.append(loss_gradients.item())
            # Update target networks
            if num_episodes % args.target_network_update_freq == 0:
                soft_update(target_net=target_network, utility_net=utility_network, polyak=args.polyak)
                soft_update(target_net=target_mixer, utility_net=mixer, polyak=args.polyak)
        # Logging
        if num_episodes % args.log_every == 0:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            writer.add_scalar("rollout/epsilon", epsilon, step)
            if "smac" in args.env_type:
                writer.add_scalar("rollout/battle_won", np.mean(ep_stats), step)
            if len(losses) > 0:
                writer.add_scalar("train/loss", np.mean(losses), step)
                writer.add_scalar("train/grads", np.mean(gradients), step)
                losses, gradients = [], []
            ep_rewards, ep_lengths, ep_stats = [], [], []

        if num_episodes % args.eval_steps == 0 or step >= args.total_timesteps - 1:
            eval_obs, _ = eval_env.reset()
            eval_ep_reward, eval_ep_length, eval_ep_stats = [], [], []
            h_eval = None
            while eval_env.get_env_mask().any():
                env_mask = eval_env.get_env_mask()
                with torch.no_grad():
                    q_values, h_eval = utility_network(
                        torch.from_numpy(eval_obs).float().flatten(0, 1).to(device),
                        h=h_eval,
                        avail_action=torch.from_numpy(eval_env.get_avail_actions())
                        .bool()
                        .flatten(0, 1)
                        .to(device),
                    )
                actions = (
                    q_values.reshape(args.num_eval_ep, eval_env.n_agents, -1).argmax(dim=-1).cpu().numpy()
                )
                eval_obs, reward, done, truncated, infos = eval_env.step(actions)
                to_store = np.logical_and(np.logical_or(done, truncated), env_mask)
                for index in np.nonzero(to_store)[0]:
                    eval_ep_reward.append(infos[index]["episode_stats"]["r"])
                    eval_ep_length.append(infos[index]["episode_stats"]["l"])
                    if "smac" in args.env_type:
                        eval_ep_stats.append(infos[index]["battle_won"])
            writer.add_scalar("eval/ep_reward", np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length", np.mean(eval_ep_length), step)
            if "smac" in args.env_type:
                writer.add_scalar("eval/battle_won", np.mean(eval_ep_stats), step)
    # ---- Save checkpoints -------
    if args.save_model:
        checkpoint = {"utility_network": utility_network.state_dict(), "mixer": mixer.state_dict()}
        if args.normalize_obs:
            checkpoint["obs_rms"] = rms_state_dict(env.get_wrapper_attr("obs_rms"))
            state_rms = env.get_wrapper_attr("state_rms")
            if state_rms is not None:
                checkpoint["state_rms"] = rms_state_dict(state_rms)
        if args.normalize_reward:
            checkpoint["return_rms"] = rms_state_dict(env.get_wrapper_attr("return_rms"))
        torch.save(checkpoint, f"{log_dir}/agent.pt")
        with open(f"{log_dir}/args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    # ---- Close loggings and envs -------
    writer.close()
    if args.use_wnb:
        wandb.finish()
    env.close()
    eval_env.close()
