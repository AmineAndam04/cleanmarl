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
    env_type: str = "pz"
    """ pz, mamujoco ... """
    env_name: str = "multiwalker_v9"
    """ Name of the environment """
    env_family: str = "sisl"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Append the agent ID (one-hot vector) to each observation"""
    normalize_obs: bool = False
    """ Normalize the observations if True"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    max_episode_steps: int = 150
    """ Maximum steps per episode"""
    # Network
    actor_hidden_dim: int = 64
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 128
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    hyper_dim: int = 32
    """ Hidden dimension of hyper-network"""
    # Training
    total_timesteps: int = 500000
    """ Total steps in the environment during training"""
    buffer_size: int = 5000
    """ The number of episodes in the replay buffer"""
    batch_size: int = 10
    """ Number of sampled episodes"""
    minibatch_size: int = 6
    """ Mini Batch size"""
    train_freq: int = 1
    """ Train every train_freq episodes"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.00001
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.00001
    """ Learning rate for the critic"""
    gamma: float = 0.99
    """ Discount factor"""
    clip_gradients: float = -1
    """ Disable gradient clipping when <= 0; otherwise clip at this value"""
    target_network_update_freq: int = 1
    """ Update the target networks every target_network_update_freq episodes"""
    polyak: float = 0.005
    """ Polyak coefficient for target network update"""
    device: str = "cpu"
    """ Device (cpu, cuda, mps)"""
    seed: int = 1
    """ Random seed"""
    # logging
    work_dir: str = "runs"
    """ Folder to save logs, weights ..."""
    save_model: bool = False
    """ If True, save the weights of the agents and hyperparameters"""
    exp_name: str = "v1"
    """ Used for logging"""
    log_every: int = 10
    """ Number of completed episodes accumulated before logging """
    eval_steps: int = 50
    """ Evaluate the policy every eval_steps episodes"""
    num_eval_ep: int = 5
    """ Number of evaluation episodes"""
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for _ in range(num_layer):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))

    def act(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.tanh(x)


class Qnetwrok(nn.Module):
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
        tot_length = sum(lengths)
        obs = torch.zeros(tot_length, self.num_agents, self.obs_space).float().to(self.device)
        next_obs = torch.zeros(tot_length, self.num_agents, self.obs_space).float().to(self.device)
        actions = torch.zeros(tot_length, self.num_agents, self.action_space).float().to(self.device)
        rewards = torch.zeros(tot_length).float().to(self.device)
        states = torch.zeros(tot_length, self.state_space).float().to(self.device)
        next_states = torch.zeros(tot_length, self.state_space).float().to(self.device)
        done = torch.ones(tot_length).int().to(self.device)
        position = 0
        for episode, length in zip(batch, lengths):
            obs[position : position + length] = episode["obs"][:-1]
            next_obs[position : position + length] = episode["obs"][1:]
            actions[position : position + length] = episode["actions"]
            rewards[position : position + length] = episode["reward"]
            states[position : position + length] = episode["states"][:-1]
            next_states[position : position + length] = episode["states"][1:]
            done[position : position + length] = episode["done"]
            position += length
        return obs, next_obs, actions, rewards, states, next_states, done


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
        elif args.env_type == "mamujoco":
            from marl_envs import MAmujocoInterface  # noqa: PLC0415

            env = MAmujocoInterface(
                env_name=args.env_name, max_episode_steps=args.max_episode_steps, **kwargs
            )
        else:
            raise ValueError(f"{args.env_type} not supported for this FACMAC")

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


def soft_update(target_net, utility_net, polyak):
    for target_param, param in zip(target_net.parameters(), utility_net.parameters()):
        target_param.data.copy_(polyak * param.data + (1.0 - polyak) * target_param.data)


def rms_state_dict(rms):
    return {"mean": torch.as_tensor(rms.mean).cpu().clone(), "var": torch.as_tensor(rms.var).cpu().clone()}


if __name__ == "__main__":
    # ---- Prepare for training: seed, networks, optim ... -------
    args = tyro.cli(Args)
    # Set the random seed
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
    actor = Actor(
        input_dim=env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=env.get_action_size(),
    ).to(device)
    target_actor = copy.deepcopy(actor).to(device)
    critic = Qnetwrok(
        input_dim=env.get_obs_size() + env.get_action_size(),
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
    ).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    mixer = MixingNetwork(n_agents=env.n_agents, s_dim=env.get_state_size(), hidden_dim=args.hyper_dim).to(
        device
    )
    target_mixer = copy.deepcopy(mixer).to(device)
    # Initialize the optimizer
    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(
        list(critic.parameters()) + list(mixer.parameters()), lr=args.learning_rate_critic
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
            name=f"FACMAC-continuous-{run_name}",
        )
    log_dir = f"{args.work_dir}/FACMAC-continuous-{run_name}"
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )
    step, num_episodes = 0, 0
    ep_rewards, ep_lengths = [], []
    cr_losses, cr_gradients, ac_losses, ac_gradients = [], [], [], []
    while step < args.total_timesteps:
        # ---- Collect an episode -------
        episode = {"obs": [], "actions": [], "reward": [], "states": [], "done": []}
        obs, _ = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            state = env.get_state()
            with torch.no_grad():
                actions = actor.act(torch.from_numpy(obs).float().to(device))
                noise = 0.05 * torch.randn_like(actions)
                actions = torch.clamp(actions + noise, -1, 1).cpu().numpy()
            next_obs, reward, done, truncated, infos = env.step(actions)
            step += 1
            episode["obs"].append(obs)
            episode["actions"].append(actions)
            episode["reward"].append(reward)
            episode["done"].append(done or truncated)
            episode["states"].append(state)
            obs = next_obs
        # Store last step
        episode["obs"].append(obs)
        episode["states"].append(env.get_state())
        rb.store(episode)
        num_episodes += 1
        ep_rewards.append(infos["episode_stats"]["r"])
        ep_lengths.append(infos["episode_stats"]["l"])
        # ---- Training loop -------
        if num_episodes > args.batch_size:
            if num_episodes % args.train_freq == 0:
                # Sample a batch of episodes
                (
                    b_obs,
                    b_next_obs,
                    b_actions,
                    b_reward,
                    b_states,
                    b_next_states,
                    b_done,
                ) = rb.sample(args.batch_size)
                # Update the actor and critic
                num_samples = b_obs.size(0)
                ac_loss, cr_loss = 0, 0
                critic_optimizer.zero_grad()
                actor_optimizer.zero_grad()
                ## Critic loss
                for start in range(0, b_obs.size(0), args.minibatch_size):
                    end = start + args.minibatch_size
                    with torch.no_grad():
                        actions_from_target_actor = target_actor.act(b_next_obs[start:end])
                        qvals_from_target_utility = target_critic(
                            torch.cat((b_next_obs[start:end], actions_from_target_actor), dim=-1)
                        )
                        q_tot_from_target_mixer = target_mixer(
                            Q=qvals_from_target_utility, s=b_next_states[start:end]
                        ).reshape(-1)
                        targets = (
                            b_reward[start:end]
                            + args.gamma * (1 - b_done[start:end]) * q_tot_from_target_mixer
                        )
                    q_values = critic(torch.cat((b_obs[start:end], b_actions[start:end]), dim=-1))
                    q_tot = mixer(Q=q_values, s=b_states[start:end]).reshape(-1)
                    critic_loss = F.mse_loss(targets, q_tot, reduction="sum") / num_samples
                    cr_loss += critic_loss.detach()
                    critic_loss.backward()
                critic_gradient = norm_d([p.grad for p in critic.parameters()], 2)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.clip_gradients)
                    torch.nn.utils.clip_grad_norm_(mixer.parameters(), max_norm=args.clip_gradients)
                critic_optimizer.step()
                ## Actor loss
                for start in range(0, b_obs.size(0), args.minibatch_size):
                    end = start + args.minibatch_size
                    actions = actor.act(b_obs[start:end])
                    q_values = critic(torch.cat((b_obs[start:end], actions), dim=-1))
                    q_tot = mixer(Q=q_values, s=b_states[start:end])
                    actor_loss = -q_tot.sum() / num_samples
                    ac_loss += actor_loss.detach()
                    actor_loss.backward()
                actor_gradient = norm_d([p.grad for p in actor.parameters()], 2)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.clip_gradients)
                actor_optimizer.step()
                cr_losses.append(cr_loss.item())
                cr_gradients.append(critic_gradient.item())
                ac_losses.append(ac_loss.item())
                ac_gradients.append(actor_gradient.item())
            # Update target networks
            if num_episodes % args.target_network_update_freq == 0:
                soft_update(target_net=target_actor, utility_net=actor, polyak=args.polyak)
                soft_update(target_net=target_critic, utility_net=critic, polyak=args.polyak)
                soft_update(target_net=target_mixer, utility_net=mixer, polyak=args.polyak)
        # Logging
        if num_episodes % args.log_every == 0:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            ep_rewards, ep_lengths = [], []
            if len(ac_losses) > 0:
                writer.add_scalar("train/critic_loss", np.mean(cr_losses), step)
                writer.add_scalar("train/actor_loss", np.mean(ac_losses), step)
                writer.add_scalar("train/ac_gradients", np.mean(ac_gradients), step)
                writer.add_scalar("train/cr_gradients", np.mean(cr_gradients), step)
                cr_losses, cr_gradients, ac_losses, ac_gradients = [], [], [], []
        # ---- Evaluate on separate envs -------
        if num_episodes % args.eval_steps == 0 or step >= args.total_timesteps - 1:
            eval_obs, _ = eval_env.reset()
            eval_ep_reward, eval_ep_length = [], []
            while eval_env.get_env_mask().any():
                env_mask = eval_env.get_env_mask()
                with torch.no_grad():
                    actions = actor.act(torch.from_numpy(eval_obs).float().to(device))
                eval_obs, reward, done, truncated, infos = eval_env.step(actions.cpu().numpy())
                to_store = np.logical_and(np.logical_or(done, truncated), env_mask)
                for index in np.nonzero(to_store)[0]:
                    eval_ep_reward.append(infos[index]["episode_stats"]["r"])
                    eval_ep_length.append(infos[index]["episode_stats"]["l"])
            writer.add_scalar("eval/ep_reward", np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length", np.mean(eval_ep_length), step)
    # ---- Save checkpoints -------
    if args.save_model:
        checkpoint = {"actor": actor.state_dict(), "critic": critic.state_dict(), "mixer": mixer.state_dict()}
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
    # ---- Close loggers and environments -------
    writer.close()
    if args.use_wnb:
        wandb.finish()
    env.close()
    eval_env.close()
