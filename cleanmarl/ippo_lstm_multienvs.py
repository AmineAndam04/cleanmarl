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
from marl_envs.vec_envs import SubprocVectorEnv, SyncVectorEnv
from marl_envs.wrappers import (
    AddAgentIDVec,
    NormalizeVecObservation,
    NormalizeVecReward,
    RecordEpisodeStatistics,
)
from torch.distributions.categorical import Categorical
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
    use_subproc: bool = True
    """ If true, put each env in a process, if not run n_episodes env in sequence"""
    agent_ids: bool = True
    """ Append the agent ID (one-hot vector) to each observation"""
    normalize_obs: bool = False
    """ Normalize the observations if True"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    max_episode_steps: int = 150
    """ Maximum steps per episode"""
    # Network
    actor_hidden_dim: int = 32
    """ Hidden dimension of actor network"""
    critic_hidden_dim: int = 32
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    # Training
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    n_episodes: int = 3
    """ Number of episodes to collect in each rollout"""
    tbptt: int = 10
    """ Chunk size for Truncated Backpropagation Through Time tbptt"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0008
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0008
    """ Learning rate for the critic"""
    ppo_clip: float = 0.2
    """ PPO clipping factor """
    entropy_coef: float = 0.001
    """ Entropy coefficient """
    epochs: int = 3
    """ Number of training epochs"""
    gamma: float = 0.99
    """ Discount factor"""
    td_lambda: float = 0.95
    """ TD(λ) parameter"""
    normalize_advantage: bool = False
    """ Normalize the advantage if True"""
    normalize_return: bool = False
    """ Normalize the returns if True"""
    clip_gradients: float = -1
    """ Disable gradient clipping when <= 0; otherwise clip at this value"""
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
    """ Number of completed episodes accumulated before logging """
    eval_steps: int = 10
    """ Evaluate the policy every eval_steps episodes"""
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""


class RolloutBuffer:
    def __init__(
        self,
        buffer_size,
        num_agents,
        obs_space,
        action_space,
        gamma,
        td_lambda,
        normalize_return,
        normalize_advantage,
        device="cpu",
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.normalize_return = normalize_return
        self.normalize_advantage = normalize_advantage
        self.device = device
        self.episodes = [None] * buffer_size
        self.pos = 0

    def add(self, episode):
        for key, values in episode.items():
            episode[key] = torch.from_numpy(np.stack(values))
        self.compute_advantage_and_return(episode=episode)
        self.episodes[self.pos] = episode
        self.pos += 1

    def compute_advantage_and_return(self, episode):
        return_lambda = torch.zeros_like(episode["actions"]).float()
        advantages = torch.zeros_like(episode["actions"]).float()
        with torch.no_grad():
            ep_len = episode["obs"].size(0)
            last_return_lambda = 0
            for t in reversed(range(ep_len)):
                next_value = 0 if t == ep_len - 1 else episode["values"][t + 1]
                return_lambda[t] = last_return_lambda = episode["reward"][t] + self.gamma * (
                    self.td_lambda * last_return_lambda + (1 - self.td_lambda) * next_value
                )
                advantages[t] = return_lambda[t] - episode["values"][t]
        episode["returns"] = return_lambda
        episode["advantages"] = advantages
        del episode["values"]
        del episode["reward"]

    def get_batch(self):
        lengths = [len(episode["obs"]) for episode in self.episodes]
        max_length = max(lengths)
        obs = (
            torch.zeros(self.buffer_size, max_length, self.num_agents, self.obs_space).float().to(self.device)
        )
        avail_actions = (
            torch.zeros(self.buffer_size, max_length, self.num_agents, self.action_space)
            .bool()
            .to(self.device)
        )
        actions = torch.zeros(self.buffer_size, max_length, self.num_agents).int().to(self.device)
        log_probs = torch.zeros(self.buffer_size, max_length, self.num_agents).float().to(self.device)
        returns = torch.zeros(self.buffer_size, max_length, self.num_agents).float().to(self.device)
        advantages = torch.zeros(self.buffer_size, max_length, self.num_agents).float().to(self.device)
        mask = torch.zeros(self.buffer_size, max_length, dtype=torch.bool).to(self.device)
        for i in range(self.buffer_size):
            length = lengths[i]
            obs[i, :length] = self.episodes[i]["obs"]
            avail_actions[i, :length] = self.episodes[i]["avail_actions"]
            actions[i, :length] = self.episodes[i]["actions"]
            log_probs[i, :length] = self.episodes[i]["log_prob"]
            returns[i, :length] = self.episodes[i]["returns"]
            advantages[i, :length] = self.episodes[i]["advantages"]
            mask[i, :length] = 1
        if self.normalize_advantage:
            advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)
        if self.normalize_return:
            returns = (returns - returns[mask].mean()) / (returns[mask].std() + 1e-8)
        self.episodes = [None] * self.buffer_size
        self.pos = 0
        return (
            obs.permute(0, 2, 1, 3).flatten(0, 1),
            actions.permute(0, 2, 1).flatten(0, 1),
            avail_actions.permute(0, 2, 1, 3).flatten(0, 1),
            log_probs.permute(0, 2, 1).flatten(0, 1),
            returns.permute(0, 2, 1).flatten(0, 1),
            advantages.permute(0, 2, 1).flatten(0, 1),
            mask.unsqueeze(-1).expand(-1, -1, self.num_agents).permute(0, 2, 1).flatten(0, 1),
        )


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc2 = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def act(self, x, h=None, avail_action=None):
        logits, h = self.logits(x, h, avail_action)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return action, distribution.log_prob(action), h

    def logits(self, x, h=None, avail_action=None):
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
            x = x.masked_fill(~avail_action, -1e9)
        return x, h

    def get_logprob_entropy(self, obs, h, action, avail_action):
        logits, h = self.logits(obs, h, avail_action)
        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(action)
        entropy = distribution.entropy()
        return log_probs, entropy, h


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for _ in range(num_layer):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, 1)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)


def make_env(args, kwargs):
    def env_fn():
        if args.env_type == "pz":
            from marl_envs import PettingZooInterface  # noqa: PLC0415

            env = PettingZooInterface(
                family=args.env_family,
                env_name=args.env_name,
                max_episode_steps=args.max_episode_steps,
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
            raise ValueError(f"{args.env_type} not supported for this IPPO")

        return RecordEpisodeStatistics(env)

    return env_fn


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.stack(norms), d)
    return total_norm_d


def rms_state_dict(rms):
    return {"mean": torch.as_tensor(rms.mean).cpu().clone(), "var": torch.as_tensor(rms.var).cpu().clone()}


if __name__ == "__main__":
    # ---- Prepare for training: seed, networks, optim ... -------
    args = tyro.cli(Args)
    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Set device
    device = torch.device(args.device)
    # Set the environment
    env_fn = make_env(args, kwargs={})
    env_parallelizer = SubprocVectorEnv if args.use_subproc else SyncVectorEnv
    envs = env_parallelizer(env_fns=[env_fn for _ in range(args.n_episodes)], auto_reset=False)
    eval_env = SyncVectorEnv(env_fns=[env_fn for _ in range(args.num_eval_ep)], auto_reset=False)
    if args.normalize_obs:
        envs = NormalizeVecObservation(envs)
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
    # Initialize the actor and the critic
    actor = Actor(
        input_dim=envs.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        output_dim=envs.get_action_size(),
    ).to(device)
    critic = Critic(
        input_dim=envs.get_obs_size(),
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
    ).to(device)
    # Initialize the optimizer
    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic)
    # Initialize the rollout buffer
    rb = RolloutBuffer(
        buffer_size=args.n_episodes,
        obs_space=envs.get_obs_size(),
        action_space=envs.get_action_size(),
        num_agents=envs.n_agents,
        gamma=args.gamma,
        td_lambda=args.td_lambda,
        normalize_return=args.normalize_return,
        normalize_advantage=args.normalize_advantage,
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
            name=f"IPPO-lstm-multienvs-{run_name}",
        )
    log_dir = f"{args.work_dir}/IPPO-lstm-multienvs-{run_name}"
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )
    step, num_episodes = 0, 0
    kl_divs, clipped_ratios = [], []
    ac_gradients, cr_gradients = [], []
    ac_losses, cr_losses, entropies = [], [], []
    ep_rewards, ep_lengths, ep_stats = [], [], []
    while step < args.total_timesteps:
        episodes = [
            {
                "obs": [],
                "actions": [],
                "log_prob": [],
                "reward": [],
                "avail_actions": [],
                "values": [],
            }
            for _ in range(args.n_episodes)
        ]
        obs, _ = envs.reset()
        h = None
        while envs.get_env_mask().any():
            env_mask = envs.get_env_mask()
            avail_action = envs.get_avail_actions()
            with torch.no_grad():
                actions, log_probs, h = actor.act(
                    torch.from_numpy(obs).float().flatten(0, 1).to(device),
                    h=h,
                    avail_action=torch.tensor(avail_action).flatten(0, 1).bool().to(device),
                )
                values = critic(torch.from_numpy(obs).float().to(device)).cpu()
                actions = actions.reshape(args.n_episodes, envs.n_agents).cpu().numpy()
                log_probs = log_probs.reshape(args.n_episodes, envs.n_agents).cpu().numpy()
            # Step the environment
            next_obs, reward, done, truncated, infos = envs.step(actions)
            step += env_mask.sum()
            for i in np.nonzero(env_mask)[0]:
                episodes[i]["obs"].append(obs[i])
                episodes[i]["actions"].append(actions[i])
                episodes[i]["log_prob"].append(log_probs[i])
                episodes[i]["reward"].append(reward[i])
                episodes[i]["avail_actions"].append(avail_action[i])
                episodes[i]["values"].append(values[i])
            obs = next_obs
            to_store = np.logical_and(np.logical_or(done, truncated), env_mask)
            for index in np.nonzero(to_store)[0]:
                rb.add(episodes[index].copy())
                ep_rewards.append(infos[index]["episode_stats"]["r"])
                ep_lengths.append(infos[index]["episode_stats"]["l"])
                if "smac" in args.env_type:
                    ep_stats.append(infos[index].get("battle_won", False))
        num_episodes += args.n_episodes
        # ---- Training loop -------
        ## Prepare the batch
        b_obs, b_actions, b_avail_actions, b_log_probs, b_returns, b_advantages, b_mask = rb.get_batch()
        for _ in range(args.epochs):
            num_samples = b_mask.sum()
            ac_loss, cr_loss, entropy, kl_div, clipped_ratio = 0, 0, 0, 0, 0
            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()
            h = None
            for start in range(0, b_obs.size(1), args.tbptt):
                end = start + args.tbptt
                current_logprob, entropy_loss, h = actor.get_logprob_entropy(
                    obs=b_obs[:, start:end],
                    h=h,
                    action=b_actions[:, start:end],
                    avail_action=b_avail_actions[:, start:end],
                )
                log_ratio = current_logprob - b_log_probs[:, start:end]
                ratio = torch.exp(log_ratio)
                pg_loss1 = b_advantages[:, start:end] * ratio
                pg_loss2 = b_advantages[:, start:end] * torch.clamp(
                    ratio, 1 - args.ppo_clip, 1 + args.ppo_clip
                )
                pg_loss = torch.min(pg_loss1[b_mask[:, start:end]], pg_loss2[b_mask[:, start:end]]).sum()
                entropy_loss = entropy_loss[b_mask[:, start:end]].sum()
                actor_loss = -pg_loss - args.entropy_coef * entropy_loss
                actor_loss /= num_samples
                actor_loss.backward()
                ac_loss += actor_loss.detach()
                entropy += (entropy_loss / num_samples).detach()
                h = (h[0].detach(), h[1].detach())
                # Critic loss
                current_values = critic(x=b_obs[:, start:end])
                critic_loss = F.mse_loss(
                    current_values[b_mask[:, start:end]],
                    b_returns[:, start:end][b_mask[:, start:end]],
                    reduction="sum",
                )
                critic_loss = critic_loss / num_samples
                cr_loss += critic_loss.detach()
                critic_loss.backward()
                # track kl distance
                with torch.no_grad():
                    b_kl_divergence = ((ratio - 1) - log_ratio)[b_mask[:, start:end]].sum()
                    kl_div += b_kl_divergence / num_samples
                    clipped_ratio += ((ratio - 1.0).abs() > args.ppo_clip)[
                        b_mask[:, start:end]
                    ].sum() / num_samples
            critic_gradient = norm_d([p.grad for p in critic.parameters()], 2)
            actor_gradient = norm_d([p.grad for p in actor.parameters()], 2)
            if args.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.clip_gradients)
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.clip_gradients)
            critic_optimizer.step()
            actor_optimizer.step()
            cr_losses.append(cr_loss.item())
            cr_gradients.append(critic_gradient.item())
            entropies.append(entropy.item())
            ac_losses.append(ac_loss.item())
            ac_gradients.append(actor_gradient.item())
            clipped_ratios.append(clipped_ratio.cpu())
            kl_divs.append(kl_div.item())
        # logging
        if len(ep_rewards) >= args.log_every:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            if "smac" in args.env_type:
                writer.add_scalar("rollout/battle_won", np.mean(ep_stats), step)
            ep_rewards, ep_lengths, ep_stats = [], [], []
            if len(ac_losses) > 0:
                writer.add_scalar("train/critic_loss", np.mean(cr_losses), step)
                writer.add_scalar("train/actor_loss", np.mean(ac_losses), step)
                writer.add_scalar("train/entropy", np.mean(entropies), step)
                writer.add_scalar("train/kl_divergence", np.mean(kl_divs), step)
                writer.add_scalar("train/clipped_ratios", np.mean(clipped_ratios), step)
                writer.add_scalar("train/ac_gradients", np.mean(ac_gradients), step)
                writer.add_scalar("train/cr_gradients", np.mean(cr_gradients), step)
                ac_losses, cr_losses, entropies = [], [], []
                ac_gradients, cr_gradients = [], []
                kl_divs, clipped_ratios = [], []
        # ---- Evaluate on separate envs -------
        if num_episodes % args.eval_steps == 0 or step >= args.total_timesteps - 1:
            eval_obs, _ = eval_env.reset()
            eval_ep_reward, eval_ep_length, eval_ep_stats = [], [], []
            h_eval = None
            while eval_env.get_env_mask().any():
                env_mask = eval_env.get_env_mask()
                with torch.no_grad():
                    logits, h_eval = actor.logits(
                        torch.from_numpy(eval_obs).float().flatten(0, 1).to(device),
                        h=h_eval,
                        avail_action=torch.from_numpy(eval_env.get_avail_actions())
                        .bool()
                        .flatten(0, 1)
                        .to(device),
                    )
                    actions = logits.reshape(args.num_eval_ep, eval_env.n_agents, -1).argmax(-1).cpu().numpy()
                eval_obs, reward, done, truncated, infos = eval_env.step(actions)
                to_store = np.logical_and(np.logical_or(done, truncated), env_mask)
                for index in np.nonzero(to_store)[0]:
                    eval_ep_reward.append(infos[index]["episode_stats"]["r"])
                    eval_ep_length.append(infos[index]["episode_stats"]["l"])
                    if "smac" in args.env_type:
                        eval_ep_stats.append(infos[index].get("battle_won", False))
            writer.add_scalar("eval/ep_reward", np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length", np.mean(eval_ep_length), step)
            if "smac" in args.env_type:
                writer.add_scalar("eval/battle_won", np.mean(eval_ep_stats), step)
    # ---- Save checkpoints -------
    if args.save_model:
        checkpoint = {"actor": actor.state_dict(), "critic": critic.state_dict()}
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
