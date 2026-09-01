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
from marlbench.vec_envs import SubprocVectorEnv, SyncVectorEnv
from marlbench.wrappers import (
    AddAgentIDVec,
    NormalizeVecObservation,
    NormalizeVecReward,
    RecordEpisodeStatistics,
)
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
    """Append the agent ID (one-hot vector) to each observation"""
    num_envs: int = 4
    """ Number of parallel environments"""
    use_subproc: bool = True
    """ Run each environment in a separate process when True; otherwise run them sequentially"""
    normalize_obs: bool = False
    """ Normalize the observations if True"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    max_episode_steps: int = 150
    """ Maximum steps per episode"""
    # Network
    hidden_dim: int = 64
    """ Hidden dimension"""
    num_layers: int = 1
    """ Number of hidden layers"""
    # Training
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    train_freq: int = 2
    """ Train every train_freq*num_envs episodes"""
    buffer_size: int = 5000
    """ The number of episodes in the replay buffer"""
    batch_size: int = 32
    """ Number of sampled episodes"""
    minibatch_size: int = 6
    """ Mini Batch size"""
    n_epochs: int = 2
    """ Number of batches sampled in one update"""
    gamma: float = 0.99
    """ Discount factor"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate: float = 0.0005
    """ Learning rate"""
    target_network_update_freq: int = 1
    """ Update the target networks every target_network_update_freq*num_envs episodes"""
    polyak: float = 0.005
    """ Polyak coefficient for target network update"""
    clip_gradients: float = 5
    """ Disable gradient clipping when <= 0; otherwise clip at this value"""
    start_e: float = 1
    """ The starting value of epsilon, for exploration"""
    end_e: float = 0.05
    """ The end value of epsilon, for exploration"""
    exploration_fraction: float = 0.05
    """ Fraction of total_timesteps over which epsilon decreases from start_e to end_e"""
    device: str = "cpu"
    """ Device (cpu, cuda, mps)"""
    seed: int = 1
    """ Random seed"""
    # Logging
    work_dir: str = "runs"
    """ Folder to save logs, weights..."""
    save_model: bool = False
    """ If True, save the weights of the agents and hyperparameters"""
    exp_name: str = "v1"
    """ Used for logging"""
    log_every: int = 10
    """ Log rollout stats every log_every*num_envs episodes"""
    eval_steps: int = 50
    """ Evaluate the policy every eval_steps*num_envs episodes"""
    num_eval_ep: int = 5
    """ Number of evaluation episodes"""
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""


class Qnetwrok(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for _ in range(num_layer):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))

    def forward(self, x, avail_action=None):
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, -1e8)
        return x


class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        num_agents,
        obs_space,
        action_space,
        device="cpu",
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.device = device
        self.episodes = [None] * buffer_size
        self.pos = 0
        self.size = 0
        ## OOM Memory issues
        self.store_type = {
            "obs": torch.float32,
            "actions": torch.int64,
            "reward": torch.float32,
            "done": torch.bool,
            "avail_actions": torch.bool,
        }

    def store(self, episode):
        for key, values in episode.items():
            vals = torch.from_numpy(np.stack(values))
            vals = vals.to(self.store_type[key])
            episode[key] = vals
        self.episodes[self.pos] = episode
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        batch = [self.episodes[i] for i in indices]
        lengths = [len(episode["obs"]) - 1 for episode in batch]
        tot_length = sum(lengths)
        obs = torch.zeros(tot_length, self.num_agents, self.obs_space).float().to(self.device)
        actions = torch.zeros(tot_length, self.num_agents).int().to(self.device)
        rewards = torch.zeros(tot_length).float().to(self.device)
        next_obs = torch.zeros(tot_length, self.num_agents, self.obs_space).float().to(self.device)
        next_avail_actions = (
            torch.zeros(tot_length, self.num_agents, self.action_space).bool().to(self.device)
        )

        done = torch.zeros(tot_length).int().to(self.device)
        position = 0
        for episode, length in zip(batch, lengths):
            obs[position : position + length] = episode["obs"][:-1]
            actions[position : position + length] = episode["actions"]
            rewards[position : position + length] = episode["reward"]
            next_obs[position : position + length] = episode["obs"][1:]
            next_avail_actions[position : position + length] = episode["avail_actions"][1:]
            done[position : position + length] = episode["done"]
            position += length
        return obs, actions, rewards, next_obs, next_avail_actions, done


def make_env(args, kwargs):
    def env_fn():
        if args.env_type == "pz":
            from marlbench import PettingZooInterface  # noqa: PLC0415

            env = PettingZooInterface(
                family=args.env_family,
                env_name=args.env_name,
                max_episode_steps=args.max_episode_steps,
                **kwargs,
            )
        elif args.env_type == "smaclite":
            from marlbench import SMACliteInterface  # noqa: PLC0415

            env = SMACliteInterface(
                env_name=args.env_name, max_episode_steps=args.max_episode_steps, **kwargs
            )
        elif args.env_type == "lbf":
            from marlbench import LBFInterface  # noqa: PLC0415

            env = LBFInterface(env_name=args.env_name, max_episode_steps=args.max_episode_steps, **kwargs)
        elif args.env_type == "rware":
            from marlbench import RWAREInterface  # noqa: PLC0415

            env = RWAREInterface(env_name=args.env_name, max_episode_steps=args.max_episode_steps, **kwargs)
        elif args.env_type == "smac":
            from marlbench.wrappers import TimeLimit  # noqa: I001, PLC0415
            from marlbench import SMACInterface  # noqa: PLC0415

            env = SMACInterface(env_name=args.env_name, seed=args.seed, **kwargs)
            env = TimeLimit(
                env=env,
                max_episode_steps=args.max_episode_steps,
            )
        elif args.env_type == "smacv2":
            from marlbench.wrappers import TimeLimit  # noqa: I001, PLC0415
            from marlbench import SMACv2Interface  # noqa: PLC0415

            env = SMACv2Interface(env_name=args.env_name, seed=args.seed, **kwargs)
            env = TimeLimit(
                env=env,
                max_episode_steps=args.max_episode_steps,
            )
        else:
            raise ValueError(f"{args.env_type} not supported for VDN")

        return RecordEpisodeStatistics(env)

    return env_fn


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.stack(norms), d)
    return total_norm_d


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
    # Set the device
    device = torch.device(args.device)
    # Set the environment
    env_fn = make_env(args, kwargs={})
    env_parallelizer = SubprocVectorEnv if args.use_subproc else SyncVectorEnv
    envs = env_parallelizer(env_fns=[env_fn for _ in range(args.num_envs)], auto_reset=False)
    eval_env = SyncVectorEnv(env_fns=[env_fn for _ in range(args.num_eval_ep)], auto_reset=False)
    if args.normalize_obs:
        envs = NormalizeVecObservation(envs, normalize_state=False)
        eval_env = NormalizeVecObservation(eval_env)
        eval_env.set_wrapper_attr("update_running_mean", False)
        eval_env.set_wrapper_attr("obs_rms", envs.get_wrapper_attr("obs_rms"))
    if args.normalize_reward:
        envs = NormalizeVecReward(envs, gamma=args.gamma)
    if args.agent_ids:
        envs = AddAgentIDVec(envs)
        eval_env = AddAgentIDVec(eval_env)
    envs.reset(seed=seed)
    eval_env.reset(seed=seed + 100)
    # Initialize the networks
    utility_network = Qnetwrok(
        input_dim=envs.get_obs_size(),
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layers,
        output_dim=envs.get_action_size(),
    ).to(device)
    target_network = copy.deepcopy(utility_network).to(device)
    # Initialize the optimizer
    optimizer = getattr(optim, args.optimizer)
    optimizer = optimizer(utility_network.parameters(), lr=args.learning_rate)
    # Initialize the replay buffer
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space=envs.get_obs_size(),
        action_space=envs.get_action_size(),
        num_agents=envs.n_agents,
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
            name=f"VDN-multienvs-{run_name}",
        )
    log_dir = f"{args.work_dir}/VDN-multienvs-{run_name}"
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )

    step, num_episodes = 0, 0
    losses, gradients = [], []
    ep_rewards, ep_lengths, ep_stats = [], [], []
    while step < args.total_timesteps:
        # ---- Collect num_envs episodes -------
        episodes = [
            {"obs": [], "actions": [], "reward": [], "done": [], "avail_actions": []}
            for _ in range(args.num_envs)
        ]
        obs, _ = envs.reset()
        avail_action = envs.get_avail_actions()
        while envs.get_env_mask().any():
            env_mask = envs.get_env_mask()
            epsilon = linear_schedule(
                args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, step
            )
            with torch.no_grad():
                q_values = utility_network(
                    torch.from_numpy(obs).float().to(device),
                    torch.from_numpy(avail_action).bool().to(device),
                )
            actions = q_values.argmax(dim=-1).cpu().numpy()
            explore = np.random.random(actions.shape) < epsilon
            if explore.any():
                actions = np.where(explore, envs.sample(), actions)
            # Step the environment
            next_obs, reward, done, truncated, infos = envs.step(actions)
            for i in np.nonzero(env_mask)[0]:
                episodes[i]["obs"].append(obs[i])
                episodes[i]["actions"].append(actions[i])
                episodes[i]["reward"].append(reward[i])
                episodes[i]["done"].append(done[i] or truncated[i])
                episodes[i]["avail_actions"].append(avail_action[i])
            step += env_mask.sum()
            obs = next_obs
            avail_action = envs.get_avail_actions()
            to_store = np.logical_and(np.logical_or(done, truncated), env_mask)
            for index in np.nonzero(to_store)[0]:
                episodes[index]["obs"].append(obs[index])
                episodes[index]["avail_actions"].append(avail_action[index])
                rb.store(episodes[index].copy())
                ep_rewards.append(infos[index]["episode_stats"]["r"])
                ep_lengths.append(infos[index]["episode_stats"]["l"])
                if "smac" in args.env_type:
                    ep_stats.append(infos[index]["battle_won"])
        num_episodes += args.num_envs
        # ---- Training loop -------
        if num_episodes > args.batch_size:
            if (num_episodes // args.num_envs) % args.train_freq == 0:
                for _ in range(args.n_epochs):
                    # Sample a batch of episodes
                    b_obs, b_action, b_reward, b_next_obs, b_next_avail_action, b_done = rb.sample(
                        args.batch_size
                    )
                    # Train the networks
                    num_samples = b_obs.size(0)
                    optimizer.zero_grad()
                    loss = 0
                    for start in range(0, b_obs.size(0), args.minibatch_size):
                        end = start + args.minibatch_size
                        with torch.no_grad():
                            q_next_max, _ = target_network(
                                b_next_obs[start:end], b_next_avail_action[start:end]
                            ).max(dim=-1)
                            vdn_q_max = q_next_max.sum(dim=-1)
                            targets = b_reward[start:end] + args.gamma * (1 - b_done[start:end]) * vdn_q_max
                        q_values = torch.gather(
                            utility_network(b_obs[start:end]), dim=-1, index=b_action[start:end].unsqueeze(-1)
                        )
                        q_values = q_values.reshape_as(q_next_max)
                        vdn_q_values = q_values.sum(dim=-1)
                        mb_loss = F.mse_loss(targets, vdn_q_values, reduction="sum")
                        mb_loss /= num_samples
                        loss += mb_loss.detach()
                        mb_loss.backward()
                    grads = [p.grad for p in utility_network.parameters()]
                    vdn_gradients = norm_d(grads, 2)
                    if args.clip_gradients > 0:
                        torch.nn.utils.clip_grad_norm_(
                            utility_network.parameters(), max_norm=args.clip_gradients
                        )
                    optimizer.step()
                    losses.append(loss.item())
                    gradients.append(vdn_gradients.item())
            # Update target networks
            if (num_episodes // args.num_envs) % args.target_network_update_freq == 0:
                soft_update(target_net=target_network, utility_net=utility_network, polyak=args.polyak)
        # Logging
        if (num_episodes // args.num_envs) % args.log_every == 0:
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
        # ---- Evaluate on separate envs -------
        if (num_episodes // args.num_envs) % args.eval_steps == 0 or step >= args.total_timesteps - 1:
            eval_obs, _ = eval_env.reset()
            eval_ep_reward, eval_ep_length, eval_ep_stats = [], [], []
            while eval_env.get_env_mask().any():
                env_mask = eval_env.get_env_mask()
                with torch.no_grad():
                    q_values = utility_network(
                        x=torch.from_numpy(eval_obs).float().to(device),
                        avail_action=torch.from_numpy(eval_env.get_avail_actions()).bool().to(device),
                    )
                actions = q_values.argmax(dim=-1).cpu().numpy()
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
        checkpoint = {"utility_network": utility_network.state_dict()}
        if args.normalize_obs:
            checkpoint["obs_rms"] = rms_state_dict(envs.get_wrapper_attr("obs_rms"))
            state_rms = envs.get_wrapper_attr("state_rms")
            if state_rms is not None:
                checkpoint["state_rms"] = rms_state_dict(state_rms)
        if args.normalize_reward:
            checkpoint["return_rms"] = rms_state_dict(envs.get_wrapper_attr("return_rms"))
        torch.save(checkpoint, f"{log_dir}/agent.pt")
        with open(f"{log_dir}/args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    # ---- Close loggers and environments -------
    writer.close()
    if args.use_wnb:
        wandb.finish()
    envs.close()
    eval_env.close()
