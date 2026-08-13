import copy
import datetime
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from marl_envs.wrappers import RecordEpisodeStatistics
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    # Environment
    env_type: str = "smaclite"  # "pz"
    """ pz(for Pettingzoo), smaclite (for SMAClite), lbf (for LBF) ... """
    env_name: str = "3m"  # "simple_spread_v3" #"pursuit_v4"
    """ Name of the environment """
    env_family: str = "mpe"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    normalize_obs: bool = False
    """ NNormalize the observations if True"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    # Network
    hidden_dim: int = 64
    """ Hidden dimension"""
    num_layers: int = 1
    """ Number of layers"""
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
            x = x.masked_fill(~avail_action, float("-inf"))
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

        self.obs = np.zeros((self.buffer_size, self.num_agents, self.obs_space))
        self.action = np.zeros((self.buffer_size, self.num_agents))
        self.reward = np.zeros(self.buffer_size)
        self.next_obs = np.zeros((self.buffer_size, self.num_agents, self.obs_space))
        self.next_avail_action = np.zeros((self.buffer_size, self.num_agents, self.action_space))
        self.done = np.zeros(self.buffer_size)
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
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[indices]).float().to(self.device),
            torch.from_numpy(self.action[indices]).long().to(self.device),
            torch.from_numpy(self.reward[indices]).float().to(self.device),
            torch.from_numpy(self.next_obs[indices]).float().to(self.device),
            torch.from_numpy(self.next_avail_action[indices]).bool().to(self.device),
            torch.from_numpy(self.done[indices]).float().to(self.device),
        )


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def make_env(args, kwargs, eval=False):
    if args.env_type == "pz":
        from marl_envs import PettingZooInterface  # noqa: PLC0415

        env = PettingZooInterface(family=args.env_family, env_name=args.env_name, **kwargs)
    elif args.env_type == "smaclite":
        from marl_envs import SMACliteInterface  # noqa: PLC0415

        env = SMACliteInterface(env_name=args.env_name, **kwargs)
    elif args.env_type == "lbf":
        from marl_envs import LBFInterface  # noqa: PLC0415

        env = LBFInterface(env_name=args.env_name, **kwargs)
    elif args.env_type == "rware":
        from marl_envs import RWAREInterface  # noqa: PLC0415

        env = RWAREInterface(env_name=args.env_name, **kwargs)
    elif args.env_type == "smac":
        from marl_envs import SMACInterface  # noqa: PLC0415

        env = SMACInterface(env_name=args.env_name, seed=args.seed, **kwargs)
    elif args.env_type == "smacv2":
        from marl_envs import SMACv2Interface  # noqa: PLC0415

        env = SMACv2Interface(env_name=args.env_name, seed=args.seed, **kwargs)
    else:
        raise ValueError(f"{args.env_type} nor supported for VDN")

    if args.normalize_obs:
        from marl_envs.wrappers import NormalizeObservation  # noqa: PLC0415

        env = NormalizeObservation(env)
        if eval:
            env.update_running_mean = False
    if args.normalize_reward and not eval:
        from marl_envs.wrappers import NormalizeReward  # noqa: PLC0415

        env = NormalizeReward(env)
    if args.agent_ids:
        from marl_envs.wrappers import AddAgentID  # noqa: PLC0415

        env = AddAgentID(env)
    return RecordEpisodeStatistics(env)


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.tensor(norms), d)
    return total_norm_d


def soft_update(target_net, utility_net, polyak):
    for target_param, param in zip(target_net.parameters(), utility_net.parameters()):
        target_param.data.copy_(polyak * param.data + (1.0 - polyak) * target_param.data)


if __name__ == "__main__":
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
    env = make_env(args, kwargs={})
    eval_env = make_env(args, kwargs={}, eval=True)
    if args.normalize_obs:
        eval_env.obs_rms = env.obs_rms  # sync env and eval env
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
    writer = SummaryWriter(f"{args.work_dir}/VDN-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )

    obs, _ = env.reset(seed=seed)
    avail_action = env.get_avail_actions()
    ep_rewards, ep_lengths, ep_stats = [], [], []
    losses, gradients = [], []
    for step in range(args.total_timesteps):
        ## select actions
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            step,
        )
        if random.random() < epsilon:
            actions = env.sample()
        else:
            with torch.no_grad():
                q_values = utility_network(
                    x=torch.from_numpy(obs).float().to(device),
                    avail_action=torch.from_numpy(avail_action).bool().to(device),
                )
            actions = torch.argmax(q_values, dim=-1).cpu().numpy()
        next_obs, reward, done, truncated, infos = env.step(actions)
        # We need the next_avail_action to compute the target loss : max of Q(next_state)
        next_avail_action = env.get_avail_actions()
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

        if step > args.learning_starts:
            if step % args.train_freq == 0:
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
            if step % args.target_network_update_freq == 0:
                soft_update(
                    target_net=target_network,
                    utility_net=utility_network,
                    polyak=args.polyak,
                )

        if len(ep_rewards) > args.log_every:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            writer.add_scalar("rollout/epsilon", epsilon, step)
            if len(losses) > 0:
                writer.add_scalar("train/loss", np.mean(losses), step)
                writer.add_scalar("train/grads", np.mean(gradients), step)
            if "smac" in args.env_type:
                writer.add_scalar("rollout/battle_won", np.mean(ep_stats), step)
            ep_rewards, ep_lengths, ep_stats = [], [], []
            losses, gradients = [], []

        if (step > 0 and step % args.eval_steps == 0) or (step >= args.total_timesteps - 1):
            eval_obs, _ = eval_env.reset()
            eval_ep_reward, eval_ep_length, eval_ep_stats = [], [], []
            eval_ep = 0
            while eval_ep < args.num_eval_ep:
                with torch.no_grad():
                    q_values = utility_network(
                        x=torch.from_numpy(eval_obs).float().to(device),
                        avail_action=torch.from_numpy(eval_env.get_avail_actions()).bool().to(device),
                    )
                actions = torch.argmax(q_values, dim=-1)
                next_obs_, reward, done, truncated, infos = eval_env.step(actions.cpu().numpy())
                eval_obs = next_obs_
                if done or truncated:
                    eval_obs, _ = eval_env.reset()
                    eval_ep_reward.append(infos["episode_stats"]["r"])
                    eval_ep_length.append(infos["episode_stats"]["r"])
                    if "smac" in args.env_type:
                        eval_ep_stats.append(infos["battle_won"])
                    current_reward = 0
                    current_ep_length = 0
                    eval_ep += 1
            writer.add_scalar("eval/ep_reward", np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/std_ep_reward", np.std(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length", np.mean(eval_ep_length), step)
            if "smac" in args.env_type:
                writer.add_scalar("eval/battle_won", np.mean(eval_ep_stats), step)
    if args.save_model:
        # Save the weights
        vdn_model_path = f"{args.work_dir}/VDN-{run_name}/agent.pt"
        torch.save(utility_network.state_dict(), vdn_model_path)
        # Save the args
        import json
        from dataclasses import asdict

        vdn_args_path = f"{args.work_dir}/VDN-{run_name}/args.json"
        with open(vdn_args_path, "w") as f:
            json.dump(asdict(args), f, indent=2)

    writer.close()
    if args.use_wnb:
        wandb.finish()
    env.close()
    eval_env.close()
