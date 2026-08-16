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
    hidden_dim: int = 64
    """ Hidden dimension"""
    num_layers: int = 1
    """ Number of hidden layers"""
    # Training
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    train_freq: int = 5
    """ Train the network each «train_freq» step in the environment"""
    buffer_size: int = 10000
    """ The size of the replay buffer"""
    batch_size: int = 32
    """ Batch size"""
    gamma: float = 0.99
    """ Discount factor"""
    learning_starts: int = 5000
    """ Number of env steps to initialize the replay buffer"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate: float = 0.0005
    """ Learning rate"""
    target_network_update_freq: int = 5
    """ Update the target network each target_network_update_freq» step in the environment"""
    polyak: float = 0.005
    """ Polyak coefficient when using polyak averaging for target network update"""
    clip_gradients: float = 5
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
    eval_steps: int = 5000
    """ Evaluate the policy each «eval_steps» steps"""
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""


class Qnetwrok(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim) -> None:
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
        self.device = device
        self.obs = np.zeros((buffer_size, num_agents, obs_space), dtype=np.float32)
        self.action = np.zeros((buffer_size, num_agents), dtype=np.int32)
        self.reward = np.zeros(buffer_size, dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, num_agents, obs_space), dtype=np.float32)
        self.next_avail_action = np.zeros((buffer_size, num_agents, action_space), dtype=np.bool_)
        self.done = np.zeros(buffer_size, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def store(self, obs, action, reward, done, next_obs, next_avail_action):
        self.obs[self.pos] = obs
        self.action[self.pos] = action
        self.reward[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.next_avail_action[self.pos] = next_avail_action
        self.done[self.pos] = done
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            torch.from_numpy(self.obs[indices]).to(self.device),
            torch.from_numpy(self.action[indices]).to(self.device),
            torch.from_numpy(self.reward[indices]).to(self.device),
            torch.from_numpy(self.next_obs[indices]).to(self.device),
            torch.from_numpy(self.next_avail_action[indices]).to(self.device),
            torch.from_numpy(self.done[indices]).to(self.device),
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
    # Set the randomness seed
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
    # Initialize the utility and target networks
    utility_network = Qnetwrok(
        input_dim=env.get_obs_size(),
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layers,
        output_dim=env.get_action_size(),
    ).to(device)
    target_network = copy.deepcopy(utility_network).to(device)
    # Initialize the optimizer
    optimizer = getattr(optim, args.optimizer)
    optimizer = optimizer(utility_network.parameters(), lr=args.learning_rate)
    # Initialize a shared replay buffer
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space=env.get_obs_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        device=device,
    )
    # Logging
    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005
    run_name = f"{args.env_type}__{args.env_name}__{args.exp_name}__{time_token}"
    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"VDN-{run_name}",
        )
    log_dir = f"{args.work_dir}/VDN-{run_name}"
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )

    obs, _ = env.reset(seed=seed)
    eval_env.reset(seed=seed + 100)
    avail_action = env.get_avail_actions()
    ep_rewards, ep_lengths, ep_stats = [], [], []
    losses, gradients = [], []
    for step in range(args.total_timesteps):
        # ---- Step the environment -------
        # select actions
        epsilon = linear_schedule(
            args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, step
        )
        with torch.no_grad():
            q_values = utility_network(
                x=torch.from_numpy(obs).float().to(device),
                avail_action=torch.from_numpy(avail_action).bool().to(device),
            )
        actions = q_values.argmax(dim=-1).cpu().numpy()
        explore = np.random.random(actions.shape) < epsilon
        if explore.any():
            actions = np.where(explore, env.sample(), actions)
        # Step the environment
        next_obs, reward, done, truncated, infos = env.step(actions)
        # We need the next_avail_action to compute the target loss : max of Q(next_state)
        next_avail_action = env.get_avail_actions()
        # Store the step
        rb.store(obs, actions, reward, done, next_obs, next_avail_action)
        obs = next_obs
        avail_action = next_avail_action
        if done or truncated:
            obs, _ = env.reset()
            avail_action = env.get_avail_actions()
            ep_rewards.append(infos["episode_stats"]["r"])
            ep_lengths.append(infos["episode_stats"]["l"])
            if "smac" in args.env_type:
                ep_stats.append(infos["battle_won"])
        # ---- Training loop -------
        if step > args.learning_starts:
            if step % args.train_freq == 0:
                # Sample a batch of steps
                (
                    batch_obs,
                    batch_action,
                    batch_reward,
                    batch_next_obs,
                    batch_next_avail_action,
                    batch_done,
                ) = rb.sample(args.batch_size)
                with torch.no_grad():
                    q_next_max, _ = target_network(batch_next_obs, avail_action=batch_next_avail_action).max(
                        dim=-1
                    )
                    vdn_q_max = q_next_max.sum(dim=-1)
                    targets = batch_reward + args.gamma * (1 - batch_done) * vdn_q_max
                q_values = torch.gather(utility_network(batch_obs), dim=-1, index=batch_action.unsqueeze(-1))
                q_values = q_values.reshape_as(q_next_max)
                vdn_q_values = q_values.sum(dim=-1)
                loss = F.mse_loss(targets, vdn_q_values)
                optimizer.zero_grad()
                loss.backward()
                grads = [p.grad for p in utility_network.parameters()]
                vdn_gradients = norm_d(grads, 2)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(utility_network.parameters(), max_norm=args.clip_gradients)
                optimizer.step()
                losses.append(loss.item())
                gradients.append(vdn_gradients.item())
            # Update the target network
            if step % args.target_network_update_freq == 0:
                soft_update(target_net=target_network, utility_net=utility_network, polyak=args.polyak)
        # Logging
        if len(ep_rewards) >= args.log_every:
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
        if (step > 0 and step % args.eval_steps == 0) or (step >= args.total_timesteps - 1):
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
