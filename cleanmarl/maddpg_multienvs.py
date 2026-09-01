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
    num_envs: int = 4
    """ Number of parallel environments"""
    use_subproc: bool = True
    """ If true, put each env in a process, if not run num_envs in sequence"""
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
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 256
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    # Training
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    buffer_size: int = 5000
    """ The number of episodes in the replay buffer"""
    batch_size: int = 10
    """  Number of sampled episodes"""
    minibatch_size: int = 64
    """ Mini Batch size"""
    epochs: int = 1
    """ Number of epochs"""
    gamma: float = 0.99
    """ Discount factor"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0003
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0003
    """ Learning rate for the critic"""
    target_network_update_freq: int = 1
    """Update the target networks every target_network_update_freq episodes"""
    polyak: float = 0.005
    """ Polyak coefficient for target network update"""
    clip_gradients: float = -1
    """ Disable gradient clipping when <= 0; otherwise clip at this value"""
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


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for _ in range(num_layer):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))

    def act(self, x, avail_action=None, hard=False):
        x = self.logits(x, avail_action)
        actions = F.gumbel_softmax(logits=x, hard=hard)
        return actions

    def logits(self, x, avail_action=None):
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, -1e8)
        return x


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim, num_agents):
        super().__init__()
        self.num_agents = num_agents
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for _ in range(num_layer):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, 1)))

    def forward(self, state, actions, grad_processing=False, batch_action=None):
        x = self.maddpg_inputs(state, actions, grad_processing, batch_action)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)

    def maddpg_inputs(self, state, actions, grad_processing, batch_action):
        maddpg_inputs = torch.zeros((state.size(0), self.num_agents, self.input_dim)).to(state.device)
        maddpg_inputs[:, :, : state.size(-1)] = state.unsqueeze(1)
        oh = actions.unsqueeze(1)
        oh = oh.expand(-1, self.num_agents, -1, -1)
        oh = oh.reshape(state.size(0), self.num_agents, -1)
        if grad_processing:
            b_oh = batch_action.unsqueeze(1)
            b_oh = b_oh.expand(-1, self.num_agents, -1, -1)
            b_oh = b_oh.reshape(state.size(0), self.num_agents, -1)
            mask = torch.eye(self.num_agents).to(state.device)
            mask = mask.unsqueeze(-1).expand(-1, -1, actions.size(-1))
            mask = mask.reshape(self.num_agents, -1)
            oh = torch.where(mask.bool(), oh, b_oh)
        maddpg_inputs[:, :, state.size(-1) :] = oh
        return maddpg_inputs


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
            raise ValueError(f"{args.env_type} not supported for this MADDPG")

        return RecordEpisodeStatistics(env)

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
    # Initialize the networks
    actor = Actor(
        input_dim=envs.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=envs.get_action_size(),
    ).to(device)
    target_actor = copy.deepcopy(actor).to(device)
    maddpg_input_dim = envs.get_state_size() + envs.n_agents * envs.get_action_size()
    critic = Critic(
        input_dim=maddpg_input_dim,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        output_dim=envs.get_action_size(),
        num_agents=envs.n_agents,
    ).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    # Initialize the optimizer
    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic)
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
            name=f"MADDPG-multienvs-{run_name}",
        )
    log_dir = f"{args.work_dir}/MADDPG-multienvs-{run_name}"
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
        # ---- Collect num_envs episodes -------
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
                # Select actions
                actions = actor.act(
                    torch.from_numpy(obs).float().to(device),
                    avail_action=torch.from_numpy(avail_action).to(device),
                    hard=True,
                )  ## These are one hot-vectors
                actions = actions.cpu().numpy()
                actions_to_take = actions.argmax(-1)
            # Step the environment
            next_obs, reward, done, truncated, infos = envs.step(actions_to_take)
            step += env_mask.sum()
            for i in np.nonzero(env_mask)[0]:
                episodes[i]["obs"].append(obs[i])
                episodes[i]["actions"].append(actions[i])
                episodes[i]["reward"].append(reward[i])
                episodes[i]["done"].append(done[i] or truncated[i])
                episodes[i]["avail_actions"].append(avail_action[i])
                episodes[i]["states"].append(state[i])
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
                num_samples = b_obs.size(0) * envs.n_agents
                ac_loss, cr_loss = 0, 0
                critic_optimizer.zero_grad()
                actor_optimizer.zero_grad()
                for start in range(0, b_obs.size(0), args.minibatch_size):
                    end = start + args.minibatch_size
                    ## Critic loss
                    with torch.no_grad():
                        actions_from_target_actor = target_actor.act(
                            b_next_obs[start:end],
                            avail_action=b_next_avail_actions[start:end],
                            hard=True,
                        )
                        qvals_from_target_critic = target_critic(
                            b_next_states[start:end], actions_from_target_actor
                        )
                        targets = (
                            b_reward[start:end].unsqueeze(1)
                            + args.gamma * (1 - b_done[start:end].unsqueeze(1)) * qvals_from_target_critic
                        )
                    q_values = critic(b_states[start:end], b_actions[start:end])
                    critic_loss = F.mse_loss(targets, q_values, reduction="sum") / num_samples
                    cr_loss += critic_loss.detach()
                    critic_loss.backward()
                critic_gradient = norm_d([p.grad for p in critic.parameters()], 2)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.clip_gradients)
                critic_optimizer.step()
                ## Actor loss
                for start in range(0, b_obs.size(0), args.minibatch_size):
                    end = start + args.minibatch_size
                    actions = actor.act(b_obs[start:end], avail_action=b_avail_actions[start:end], hard=True)
                    qvals = critic(
                        b_states[start:end], actions, grad_processing=True, batch_action=b_actions[start:end]
                    )
                    actor_loss = -qvals.sum() / num_samples
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
                        avail_action=torch.from_numpy(eval_env.get_avail_actions()).to(device),
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
