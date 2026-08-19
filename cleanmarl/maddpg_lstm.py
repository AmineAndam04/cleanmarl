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
    actor_hidden_dim: int = 32
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 2
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 128
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    # Training
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    buffer_size: int = 5000
    """ The number of episodes in the replay buffer"""
    batch_size: int = 10
    """ Batch size"""
    tbptt: int = 10
    """Chunck size for Truncated Backpropagation Through Time tbptt"""
    train_freq: int = 1
    """ Train the network each «train_freq» step in the environment"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0003
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0003
    """ Learning rate for the critic"""
    gamma: float = 0.99
    """ Discount factor"""
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    target_network_update_freq: int = 1
    """ Update the target network each target_network_update_freq» step in the environment"""
    polyak: float = 0.005
    """ Polyak coefficient when using polyak averaging for target network update"""
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
        obs = torch.zeros(batch_size, max_length, self.num_agents, self.obs_space).float().to(self.device)
        next_obs = (
            torch.zeros(batch_size, max_length, self.num_agents, self.obs_space).float().to(self.device)
        )
        avail_actions = (
            torch.zeros(batch_size, max_length, self.num_agents, self.action_space).bool().to(self.device)
        )
        next_avail_actions = (
            torch.zeros(batch_size, max_length, self.num_agents, self.action_space).bool().to(self.device)
        )
        actions = (
            torch.zeros(batch_size, max_length, self.num_agents, self.action_space).long().to(self.device)
        )
        rewards = torch.zeros(batch_size, max_length).float().to(self.device)
        states = torch.zeros(batch_size, max_length, self.state_space).float().to(self.device)
        next_states = torch.zeros(batch_size, max_length, self.state_space).float().to(self.device)
        done = torch.ones(batch_size, max_length).int().to(self.device)
        mask = torch.zeros(batch_size, max_length).bool().to(self.device)
        for i in range(batch_size):
            length = lengths[i]
            obs[i, :length] = batch[i]["obs"][:-1]
            next_obs[i, :length] = batch[i]["obs"][1:]
            avail_actions[i, :length] = batch[i]["avail_actions"][:-1]
            next_avail_actions[i, :length] = batch[i]["avail_actions"][1:]
            actions[i, :length] = batch[i]["actions"]
            rewards[i, :length] = batch[i]["reward"]
            states[i, :length] = batch[i]["states"][:-1]
            next_states[i, :length] = batch[i]["states"][1:]
            done[i, :length] = batch[i]["done"]
            mask[i, :length] = 1
        return (
            obs.permute(0, 2, 1, 3),
            next_obs.permute(0, 2, 1, 3),
            actions,
            rewards,
            states,
            next_states,
            avail_actions.permute(0, 2, 1, 3),
            next_avail_actions.permute(0, 2, 1, 3),
            done,
            mask,
        )


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc2 = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def act(self, x, h=None, avail_action=None, hard=False):
        x, h = self.logits(x, h, avail_action)
        actions = F.gumbel_softmax(logits=x, hard=hard)
        return actions, h

    def logits(self, x, h, avail_action=None):
        if h is None:
            h = (
                torch.zeros(1, x.size(0), self.hidden_dim, device=x.device),
                torch.zeros(1, x.size(0), self.hidden_dim, device=x.device),
            )
        if x.dim() < 3:
            x = x.unsqueeze(1)
            if avail_action is not None:
                avail_action = avail_action.unsqueeze(1)
        x = self.fc1(x)
        x, h = self.lstm(x, h)
        x = self.fc2(x)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, -1e8)
        return x, h


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim, num_agents) -> None:
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
        output_dim=env.get_action_size(),
    ).to(device)
    target_actor = copy.deepcopy(actor).to(device)
    maddpg_input_dim = env.get_state_size() + env.n_agents * env.get_action_size()
    critic = Critic(
        input_dim=maddpg_input_dim,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        output_dim=env.get_action_size(),
        num_agents=env.n_agents,
    ).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    # Initialize the optimizer
    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic)
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
            name=f"MADDPG-lstm--{run_name}",
        )
    log_dir = f"{args.work_dir}/MADDPG-lstm-{run_name}"
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
        # ---- Collect an episode -------
        episode = {"obs": [], "actions": [], "reward": [], "states": [], "done": [], "avail_actions": []}
        obs, _ = env.reset()
        done, truncated = False, False
        h = None
        while not done and not truncated:
            avail_action = env.get_avail_actions()
            state = env.get_state()
            with torch.no_grad():
                actions, h = actor.act(
                    torch.from_numpy(obs).float().to(device),
                    h,
                    avail_action=torch.from_numpy(avail_action).bool().to(device),
                    hard=True,
                )  ## These are one hot-vectors
                actions = actions.squeeze(1)
                actions_to_take = actions.argmax(dim=-1).cpu().numpy()
            next_obs, reward, done, truncated, infos = env.step(actions_to_take)
            step += 1
            episode["obs"].append(obs)
            episode["actions"].append(actions.cpu())
            episode["reward"].append(reward)
            episode["done"].append(done or truncated)
            episode["avail_actions"].append(avail_action)
            episode["states"].append(state)
            obs = next_obs
        # Store last step
        episode["obs"].append(obs)
        episode["states"].append(env.get_state())
        episode["avail_actions"].append(env.get_avail_actions())
        rb.store(episode)
        num_episodes += 1
        ep_rewards.append(infos["episode_stats"]["r"])
        ep_lengths.append(infos["episode_stats"]["l"])
        if "smac" in args.env_type:
            ep_stats.append(infos["battle_won"])
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
                    b_avail_actions,
                    b_next_avail_actions,
                    b_done,
                    b_mask,
                ) = rb.sample(args.batch_size)
                ## train the critic
                # Update the actor and critic
                num_samples = b_mask.sum() * env.n_agents
                ac_loss, cr_loss = 0, 0
                critic_optimizer.zero_grad()
                actor_optimizer.zero_grad()
                h_targ = None
                with torch.no_grad():
                    _, h_targ = target_actor.act(b_obs[:, :, :1].flatten(0, 1))
                for start in range(0, b_obs.size(2), args.tbptt):
                    end = start + args.tbptt
                    mb_next_obs = b_next_obs[:, :, start:end].flatten(0, 1)
                    mb_next_avail_actions = b_next_avail_actions[:, :, start:end].flatten(0, 1)
                    mb_next_states = b_next_states[:, start:end].flatten(0, 1)
                    mb_reward = b_reward[:, start:end].flatten(0, 1).unsqueeze(1)
                    mb_done = b_done[:, start:end].flatten(0, 1).unsqueeze(1)
                    mb_states = b_states[:, start:end].flatten(0, 1)
                    mb_actions = b_actions[:, start:end].flatten(0, 1)
                    mb_mask = b_mask[:, start:end].flatten(0, 1)
                    with torch.no_grad():
                        actions_from_target_actor, h_targ = target_actor.act(
                            x=mb_next_obs, h=h_targ, avail_action=mb_next_avail_actions, hard=True
                        )
                        actions_from_target_actor = actions_from_target_actor.reshape(
                            args.batch_size, env.n_agents, b_next_obs[:, :, start:end].size(2), -1
                        ).transpose(1, 2)
                        qvals_from_taget_critic = target_critic(
                            mb_next_states, actions_from_target_actor.flatten(0, 1)
                        )
                        targets = mb_reward + args.gamma * (1 - mb_done) * qvals_from_taget_critic
                    q_values = critic(mb_states, mb_actions)
                    critic_loss = F.mse_loss(
                        targets[mb_mask].reshape(-1), q_values[mb_mask].reshape(-1), reduction="sum"
                    )
                    critic_loss /= num_samples
                    cr_loss += critic_loss.detach()
                    critic_loss.backward()

                critic_gradient = norm_d([p.grad for p in critic.parameters()], 2)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.clip_gradients)
                critic_optimizer.step()
                ## Actor loss
                h = None
                for start in range(0, b_obs.size(2), args.tbptt):
                    end = start + args.tbptt
                    mb_obs = b_obs[:, :, start:end].flatten(0, 1)
                    mb_avail_actions = b_avail_actions[:, :, start:end].flatten(0, 1)
                    mb_states = b_states[:, start:end].flatten(0, 1)
                    mb_actions = b_actions[:, start:end].flatten(0, 1)
                    mb_mask = b_mask[:, start:end].flatten(0, 1)
                    actions, h = actor.act(x=mb_obs, h=h, avail_action=mb_avail_actions, hard=True)
                    actions = actions.reshape(
                        args.batch_size, env.n_agents, b_obs[:, :, start:end].size(2), -1
                    ).transpose(1, 2)

                    qvals = critic(
                        mb_states,
                        actions.flatten(0, 1),
                        grad_processing=True,
                        batch_action=mb_actions,
                    )
                    actor_loss = -qvals[mb_mask].sum() / num_samples
                    ac_loss += actor_loss.detach()
                    actor_loss.backward()
                    h = (h[0].detach(), h[1].detach())
                actor_gradient = norm_d([p.grad for p in actor.parameters()], 2)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.clip_gradients)
                actor_optimizer.step()
                cr_losses.append(cr_loss.item())
                cr_gradients.append(critic_gradient.item())
                ac_losses.append(ac_loss.item())
                ac_gradients.append(actor_gradient.item())
            # Update target actor and critic
            if num_episodes % args.target_network_update_freq == 0:
                soft_update(target_net=target_actor, utility_net=actor, polyak=args.polyak)
                soft_update(target_net=target_critic, utility_net=critic, polyak=args.polyak)
        # Logging
        if num_episodes % args.log_every == 0:
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
