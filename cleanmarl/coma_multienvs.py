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
    """ If true, put each env in a process, if not run batch_size env in sequence"""
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
    critic_hidden_dim: int = 128
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    # Training
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    batch_size: int = 3
    """ Number of episodes to collect in parallel in each rollout"""
    minibatch_size: int = 64
    """ Mini Batch size"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0005
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0005
    """ Learning rate for the critic"""
    gamma: float = 0.99
    """ Discount factor"""
    td_lambda: float = 0.8
    """ TD(λ) parameter"""
    normalize_advantage: bool = True
    """ Normalize the advantage if True"""
    target_network_update_freq: int = 1
    """ Update the target critic every target_network_update_freq training updates """
    polyak: float = 0.005
    """ Polyak coefficient when using polyak averaging for target network update"""
    entropy_coef: float = 0.001
    """ Entropy coefficient """
    use_tdlamda: bool = True
    """ Use TD(λ) as a target for the critic, if False use n-step returns (n=nsteps) """
    nsteps: int = 1
    """ Number of steps used for n-step critic targets """
    clip_gradients: float = -1
    """ Disable gradient clipping when <= 0; otherwise clip at this value"""
    start_e: float = 0.5
    """ The starting value of epsilon. See Architecture & Training in COMA's paper Sec. 5"""
    end_e: float = 0.002
    """ The end value of epsilon. See Architecture & Training in COMA's paper Sec. 5"""
    exploration_fraction: float = 750
    """ Number of training updates over which epsilon decays from start_e to end_e """
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

    def add(self, episode):
        for key, values in episode.items():
            episode[key] = torch.from_numpy(np.stack(values))
        self.compute_return(episode=episode)
        self.episodes[self.pos] = episode
        self.pos += 1

    def compute_return(self, episode):
        # 2. Compute TD(λ) using "Reconciling λ-Returns with Experience Replay"(https://arxiv.org/pdf/1810.09967 Equation 3)
        # 2. or use TD n-steps
        returns = torch.zeros_like(episode["actions"]).float()
        ep_len = episode["obs"].size(0)
        if args.use_tdlamda:
            last_returns = 0
            for t in reversed(range(ep_len)):
                next_action_value = 0 if t == ep_len - 1 else episode["values"][t + 1]
                returns[t] = last_returns = episode["reward"][t] + args.gamma * (
                    args.td_lambda * last_returns + (1 - args.td_lambda) * next_action_value
                )
        else:
            for t in range(ep_len):
                return_t_n = episode["reward"][t : t + args.nsteps]
                discounts = torch.tensor([args.gamma**i for i in range(return_t_n.size(-1))])
                return_t_n = (return_t_n * discounts).sum(-1)
                if t < (ep_len - args.nsteps):
                    return_t_n = return_t_n + args.gamma**args.nsteps * episode["values"][t + args.nsteps]
                returns[t] = return_t_n
        episode["returns"] = returns
        del episode["values"]

    def get_batch(self):
        lengths = [len(episode["obs"]) for episode in self.episodes]
        tot_length = sum(lengths)
        obs = torch.zeros(tot_length, self.num_agents, self.obs_space).float().to(self.device)
        avail_actions = torch.zeros(tot_length, self.num_agents, self.action_space).bool().to(self.device)
        actions = torch.zeros(tot_length, self.num_agents).long().to(self.device)
        rewards = torch.zeros(tot_length).float().to(self.device)
        returns = torch.zeros(tot_length, self.num_agents).float().to(self.device)
        states = torch.zeros(tot_length, self.state_space).float().to(self.device)
        position = 0
        for episode, length in zip(self.episodes, lengths):
            obs[position : position + length] = episode["obs"]
            avail_actions[position : position + length] = episode["avail_actions"]
            actions[position : position + length] = episode["actions"]
            rewards[position : position + length] = episode["reward"]
            returns[position : position + length] = episode["returns"]
            states[position : position + length] = episode["states"]
            position += length
        self.episodes = [None] * self.buffer_size
        self.pos = 0
        return obs, actions, rewards, returns, states, avail_actions


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for _ in range(num_layer):
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))

    def act(self, x, eps=0, avail_action=None):
        probs = self.logits(x=x, eps=eps, avail_action=avail_action)
        distribution = Categorical(probs)
        action = distribution.sample()
        return action

    def logits(self, x, eps=0, avail_action=None):
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, -1e9)
        masked_eps = (avail_action) * (eps / avail_action.sum(dim=-1, keepdim=True))
        probs = (1 - eps) * F.softmax(x, dim=-1) + masked_eps
        return probs


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
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))

    def forward(self, state, observations, actions, avail_actions=None):
        unbatched = state.dim() < 2
        if unbatched:
            state = state.unsqueeze(0)
            observations = observations.unsqueeze(0)
            actions = actions.unsqueeze(0)
            if avail_actions is not None:
                avail_actions = avail_actions.unsqueeze(0)
        x = self.coma_inputs(state, observations, actions)
        for layer in self.layers:
            x = layer(x)
        if avail_actions is not None:
            x = x.masked_fill(~avail_actions, -1e9)
        return x.squeeze(0) if unbatched else x

    def coma_inputs(self, state, observations, actions):
        coma_inputs = torch.zeros((state.size(0), self.num_agents, self.input_dim)).to(state.device)
        coma_inputs[:, :, : state.size(-1)] = state.unsqueeze(1)
        coma_inputs[:, :, state.size(-1) : state.size(-1) + observations.size(-1)] = observations
        one_hot = F.one_hot(actions.long(), num_classes=self.output_dim).float()
        mask = ~torch.eye(self.num_agents).bool().to(state.device)
        oh = one_hot.unsqueeze(1).expand(state.size(0), self.num_agents, self.num_agents, self.output_dim)
        oh = oh[mask.unsqueeze(0).expand(state.size(0), -1, -1)]
        oh = oh.view(state.size(0), self.num_agents, (self.num_agents - 1) * self.output_dim)
        coma_inputs[:, :, state.size(-1) + observations.size(-1) :] = oh
        return coma_inputs


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
            raise ValueError(f"{args.env_type} not supported for this COMA")

        return RecordEpisodeStatistics(env)

    return env_fn


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.stack(norms), d)
    return total_norm_d


def soft_update(target_net, critic_net, polyak):
    for target_param, param in zip(target_net.parameters(), critic_net.parameters()):
        target_param.data.copy_(polyak * param.data + (1.0 - polyak) * target_param.data)


def get_coma_critic_input_dim(env):
    critic_input_dim = env.get_obs_size() + env.get_state_size() + (env.n_agents - 1) * env.get_action_size()
    return critic_input_dim


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
    envs = env_parallelizer(env_fns=[env_fn for _ in range(args.batch_size)], auto_reset=False)
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
    # Initialize the actor, critic and target-critic networks
    actor = Actor(
        input_dim=eval_env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=eval_env.get_action_size(),
    ).to(device)
    critic_input_dim = get_coma_critic_input_dim(eval_env)
    critic = Critic(
        input_dim=critic_input_dim,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        output_dim=eval_env.get_action_size(),
        num_agents=eval_env.n_agents,
    ).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    # Initialize the optimizer
    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic)
    # Initialize the replay buffer
    rb = RolloutBuffer(
        buffer_size=args.batch_size,
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
            name=f"COMA-multienvs-{run_name}",
        )
    log_dir = f"{args.work_dir}/COMA-multienvs-{run_name}"
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )
    step = 0
    training_step = 0
    ep_rewards, ep_lengths, ep_stats = [], [], []
    cr_losses, ac_losses, entropies, ac_gradients, cr_gradients = [], [], [], [], []
    while step < args.total_timesteps:
        # ---- Collect some episodes -------
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction, training_step)
        episodes = [
            {"obs": [], "actions": [], "reward": [], "states": [], "avail_actions": [], "values": []}
            for _ in range(args.batch_size)
        ]
        obs, _ = envs.reset()
        while envs.get_env_mask().any():
            env_mask = envs.get_env_mask()
            avail_action = envs.get_avail_actions()
            state = envs.get_state()
            with torch.no_grad():
                # Select actions
                actions = actor.act(
                    torch.from_numpy(obs).float().to(device),
                    eps=epsilon,
                    avail_action=torch.from_numpy(avail_action).to(device),
                )
                # Compute its value
                values = target_critic(
                    state=torch.from_numpy(state).float().to(device),
                    observations=torch.from_numpy(obs).float().to(device),
                    actions=actions,
                    avail_actions=torch.from_numpy(avail_action).to(device),
                )
                value = torch.gather(values, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1).cpu()
                actions = actions.cpu().numpy()
            # Step the environment
            next_obs, reward, done, truncated, infos = envs.step(actions)
            step += env_mask.sum()
            for i in np.nonzero(env_mask)[0]:
                episodes[i]["obs"].append(obs[i])
                episodes[i]["actions"].append(actions[i])
                episodes[i]["reward"].append(reward[i])
                episodes[i]["avail_actions"].append(avail_action[i])
                episodes[i]["states"].append(state[i])
                episodes[i]["values"].append(value[i])
            obs = next_obs
            to_store = np.logical_and(np.logical_or(done, truncated), env_mask)
            for index in np.nonzero(to_store)[0]:
                rb.add(episodes[index].copy())
                ep_rewards.append(infos[index]["episode_stats"]["r"])
                ep_lengths.append(infos[index]["episode_stats"]["l"])
                if "smac" in args.env_type:
                    ep_stats.append(infos[index].get("battle_won", False))
        # ---- Training loop -------
        # Prepare the batch
        b_obs, b_actions, b_rewards, b_returns, b_states, b_avail_actions = rb.get_batch()
        # Update critic and actor
        num_samples = b_obs.size(0) * envs.n_agents
        ac_loss, entropy, cr_loss = 0, 0, 0
        critic_optimizer.zero_grad()
        actor_optimizer.zero_grad()
        for start in range(0, b_obs.size(0), args.minibatch_size):
            end = start + args.minibatch_size
            # Critic loss
            b_q_values = critic(
                state=b_states[start:end], observations=b_obs[start:end], actions=b_actions[start:end]
            )
            cr_values = torch.gather(b_q_values, dim=-1, index=b_actions[start:end].unsqueeze(-1))
            critic_loss = F.mse_loss(cr_values.reshape(-1), b_returns[start:end].reshape(-1), reduction="sum")
            critic_loss = critic_loss / num_samples
            cr_loss += critic_loss.detach()
            critic_loss.backward()
            # Actor loss
            pi = actor.logits(b_obs[start:end], avail_action=b_avail_actions[start:end])
            log_pi = torch.log(pi + 1e-8)
            entropy_loss = -(pi * log_pi).sum()
            ac_values = b_q_values.detach()
            coma_baseline = pi * ac_values
            coma_baseline = coma_baseline.sum(dim=-1)
            current_q = torch.gather(ac_values, dim=-1, index=b_actions[start:end].unsqueeze(-1)).squeeze(-1)
            advantage = (current_q - coma_baseline).detach()
            if args.normalize_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            log_pi = torch.gather(log_pi, dim=-1, index=b_actions[start:end].unsqueeze(-1)).squeeze(-1)
            actor_loss = (log_pi * advantage).sum()
            actor_loss = -actor_loss - args.entropy_coef * entropy_loss
            actor_loss /= num_samples
            actor_loss.backward()
            ac_loss += actor_loss.detach()
            entropy += (entropy_loss / num_samples).detach()
        critic_gradient = norm_d([p.grad for p in critic.parameters()], 2)
        actor_gradient = norm_d([p.grad for p in actor.parameters()], 2)
        if args.clip_gradients > 0:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.clip_gradients)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.clip_gradients)
        critic_optimizer.step()
        actor_optimizer.step()
        training_step += 1
        cr_losses.append(cr_loss.item())
        cr_gradients.append(critic_gradient.item())
        entropies.append(entropy.item())
        ac_losses.append(ac_loss.item())
        ac_gradients.append(actor_gradient.item())
        # Update target critic
        if training_step % args.target_network_update_freq == 0:
            soft_update(target_net=target_critic, critic_net=critic, polyak=args.polyak)
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
                writer.add_scalar("train/ac_gradients", np.mean(ac_gradients), step)
                writer.add_scalar("train/cr_gradients", np.mean(cr_gradients), step)
                cr_losses, ac_losses, entropies, ac_gradients, cr_gradients = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
        # ---- Evaluate on separate envs -------
        if training_step % args.eval_steps == 0 or step >= args.total_timesteps - 1:
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
