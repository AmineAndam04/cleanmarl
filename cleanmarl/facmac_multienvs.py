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
    num_envs: int = 4
    """ Number of parallel environments"""
    use_subproc: bool = True
    """ If true, put each env in a process, if not run num_envs in sequence"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    normalize_obs: bool = False
    """ NNormalize the observations if True"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    max_episode_steps: int = 150
    "Maximum steps per episode"
    # Network
    actor_hidden_dim: int = 32
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
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    buffer_size: int = 5000
    """ The number of episodes in the replay buffer"""
    batch_size: int = 10
    """ Batch size"""
    minibatch_size: int = 6
    """ Mini Batch size"""
    epochs: int = 1
    """ Training epochs"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0008
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0008
    """ Learning rate for the critic"""
    target_network_update_freq: int = 1
    """ Update the target network each target_network_update_freq» step in the environment"""
    polyak: float = 0.005
    """ Polyak coefficient when using polyak averaging for target network update"""
    gamma: float = 0.99
    """ Discount factor"""
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    start_e: float = 0.5
    """ The starting value of epsilon"""
    end_e: float = 0.002
    """ The end value of epsilon"""
    exploration_fraction: float = 750
    """ The number of training steps it takes from to go from start_e to  end_e"""
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


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for _ in range(num_layer):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
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
        avail_actions = torch.zeros(tot_length, self.num_agents, self.action_space).bool().to(self.device)
        next_avail_actions = (
            torch.zeros(tot_length, self.num_agents, self.action_space).bool().to(self.device)
        )
        actions = torch.zeros(tot_length, self.num_agents, self.action_space).long().to(self.device)
        rewards = torch.zeros(tot_length).float().to(self.device)
        states = torch.zeros(tot_length, self.state_space).float().to(self.device)
        next_states = torch.zeros(tot_length, self.state_space).float().to(self.device)
        done = torch.ones(tot_length).int().to(self.device)
        position = 0
        for episode, length in zip(batch, lengths):
            obs[position : position + length] = episode["obs"][:-1]
            next_obs[position : position + length] = episode["obs"][1:]
            avail_actions[position : position + length] = episode["avail_actions"][:-1]
            next_avail_actions[position : position + length] = episode["avail_actions"][1:]
            actions[position : position + length] = episode["actions"]
            rewards[position : position + length] = episode["reward"]
            states[position : position + length] = episode["states"][:-1]
            next_states[position : position + length] = episode["states"][1:]
            done[position : position + length] = episode["done"]
            position += length
        return obs, next_obs, actions, rewards, states, next_states, avail_actions, next_avail_actions, done


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
            raise ValueError(f"{args.env_type} nor supported for VDN")

        return RecordEpisodeStatistics(env)

    return env_fn


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.stack(norms), d)
    return total_norm_d


def soft_update(target_net, utility_net, polyak):
    for target_param, param in zip(target_net.parameters(), utility_net.parameters()):
        target_param.data.copy_(polyak * param.data + (1.0 - polyak) * target_param.data)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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
    # Set the environment
    env_fn = make_env(args, kwargs={})
    env_parallelizer = SubprocVectorEnv if args.use_subproc else SyncVectorEnv
    envs = env_parallelizer(env_fns=[env_fn for _ in range(args.num_envs)], auto_reset=False)
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
    # Initialize the netowrks
    actor = Actor(
        input_dim=envs.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=envs.get_action_size(),
    ).to(device)
    target_actor = copy.deepcopy(actor).to(device)
    critic = Qnetwrok(
        input_dim=envs.get_obs_size() + envs.get_action_size(),
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
    ).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    mixer = MixingNetwork(
        n_agents=envs.n_agents,
        s_dim=envs.get_state_size(),
        hidden_dim=args.hyper_dim,
    ).to(device)
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
        obs_space=envs.get_obs_size(),
        state_space=envs.get_state_size(),
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
            name=f"FACMAC-multienvs-{run_name}",
        )
    log_dir = f"{args.work_dir}/FACMAC-multienvs-{run_name}"
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )
    step, num_episodes = 0, 0
    ep_rewards, ep_lengths, ep_stats = [], [], []
    cr_losses, cr_gradients, ac_losses, ac_gradients = [], [], [], []
    while step < args.total_timesteps:
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction, num_episodes)
        episodes = [
            {"obs": [], "actions": [], "reward": [], "states": [], "done": [], "avail_actions": []}
            for _ in range(args.num_envs)
        ]
        obs, _ = envs.reset()
        avail_action = envs.get_avail_actions()
        state = envs.get_state()
        while envs.get_env_mask().any():
            env_mask = envs.get_env_mask()
            with torch.no_grad():
                actions = actor.act(
                    torch.from_numpy(obs).float().to(device),
                    eps=epsilon,
                    avail_action=torch.from_numpy(avail_action).bool().to(device),
                    hard=True,
                ).cpu()  ## These are one hot-vectors
                if epsilon > 0:
                    actions_to_take = actions.clone()
                    actions = F.one_hot(actions.long(), num_classes=envs.get_action_size())
                else:
                    actions_to_take = torch.argmax(actions, dim=-1)
                actions_to_take = actions_to_take.cpu().numpy()
            # Step the environment
            next_obs, reward, done, truncated, infos = envs.step(actions_to_take)
            step += env_mask.sum()
            for i in np.nonzero(env_mask)[0]:
                episodes[i]["obs"].append(obs[i])
                episodes[i]["actions"].append(actions[i])
                episodes[i]["reward"].append(reward[i])
                episodes[i]["states"].append(state[i])
                episodes[i]["done"].append(done[i])
                episodes[i]["avail_actions"].append(avail_action[i])

            obs = next_obs
            state = envs.get_state()
            avail_action = envs.get_avail_actions()
            to_store = np.logical_and(np.logical_or(done, truncated), env_mask)
            for index in np.nonzero(to_store)[0]:
                episodes[index]["obs"].append(obs[index])
                episodes[index]["states"].append(state[index])
                episodes[index]["avail_actions"].append(avail_action[index])
                rb.store(episodes[index].copy())
                ep_rewards.append(infos[index]["episode_stats"]["r"])
                ep_lengths.append(infos[index]["episode_stats"]["l"])
                if "smac" in args.env_type:
                    ep_stats.append(infos[index]["battle_won"])
        num_episodes += args.num_envs
        # ---- Training loop -------
        if num_episodes > args.batch_size:
            for _ in range(args.epochs):
                # Sample a batch of episodes
                (
                    b_obs,
                    b_next_obs,
                    b_actions,
                    b_reward,
                    b_states,
                    b_next_states,
                    b_avail_actions,
                    b_next_avail_actions,
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
                        actions_from_target_actor = target_actor.act(
                            b_next_obs[start:end],
                            avail_action=b_next_avail_actions[start:end],
                            hard=True,
                        )
                        qvals_from_taget_utility = target_critic(
                            torch.cat((b_next_obs[start:end], actions_from_target_actor), dim=-1)
                        )
                        q_tot_from_target_mixer = target_mixer(
                            Q=qvals_from_taget_utility, s=b_next_states[start:end]
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
                    actions = actor.act(b_obs[start:end], avail_action=b_avail_actions[start:end], hard=True)
                    q_values = critic(torch.cat((b_obs[start:end], actions), dim=-1)).squeeze()
                    q_tot = mixer(Q=q_values, s=b_states[start:end]).squeeze()
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
            # Update target actor and critic
            if (num_episodes // args.num_envs) % args.target_network_update_freq == 0:
                soft_update(target_net=target_actor, utility_net=actor, polyak=args.polyak)
                soft_update(target_net=target_critic, utility_net=critic, polyak=args.polyak)
                soft_update(target_net=target_mixer, utility_net=mixer, polyak=args.polyak)
        # Logging
        if (num_episodes // args.num_envs) % args.log_every == 0:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            if "smac" in args.env_type:
                writer.add_scalar("rollout/battle_won", np.mean(ep_stats), step)
            ep_rewards, ep_lengths, ep_stats = [], [], []
            if len(ac_losses) > 0:
                writer.add_scalar("train/critic_loss", np.mean(cr_losses), step)
                writer.add_scalar("train/actor_loss", np.mean(ac_losses), step)
                writer.add_scalar("train/ac_gradients", np.mean(ac_gradients), step)
                writer.add_scalar("train/cr_gradients", np.mean(cr_gradients), step)
                cr_losses, cr_gradients, ac_losses, ac_gradients = [], [], [], []
        # ---- Evaluate on separate envs -------
        if (num_episodes // args.num_envs) % args.eval_steps == 0 or step >= args.total_timesteps - 1:
            eval_obs, _ = eval_env.reset()
            eval_ep_reward, eval_ep_length, eval_ep_stats = [], [], []
            while eval_env.get_env_mask().any():
                env_mask = eval_env.get_env_mask()
                with torch.no_grad():
                    logits = actor.logits(
                        torch.from_numpy(eval_obs).float().to(device),
                        avail_action=torch.from_numpy(eval_env.get_avail_actions()).bool().to(device),
                    )
                    actions = logits.argmax(-1).cpu().numpy()
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
        checkpoint = {"actor": actor.state_dict(), "critic": critic.state_dict(), "mixer": mixer.state_dict()}
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
    # ---- Close loggings and envs -------
    writer.close()
    if args.use_wnb:
        wandb.finish()
    envs.close()
    eval_env.close()
