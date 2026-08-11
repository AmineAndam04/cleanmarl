import datetime
import random
from dataclasses import dataclass
from multiprocessing import Pipe, Process

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from env.lbf_wrapper import LBFWrapper
from env.pettingzoo_wrapper import PettingZooWrapper
from env.smaclite_wrapper import SMACliteWrapper
from torch.distributions.categorical import Categorical
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
    # Network
    actor_hidden_dim: int = 32
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 64
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    # Training
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    n_episodes: int = 3
    """ Number of episodes to collect in each rollout"""
    tbptt: int = 10
    """Chunck size for Truncated Backpropagation Through Time tbptt"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0008
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0008
    """ Learning rate for the critic"""
    epochs: int = 3
    """ Number of training epochs"""
    ppo_clip: float = 0.2
    """ PPO clipping factor """
    entropy_coef: float = 0.001
    """ Entropy coefficient """
    gamma: float = 0.99
    """ Discount factor"""
    td_lambda: float = 0.95
    """ TD(λ) discount factor"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    normalize_advantage: bool = False
    """ Normalize the advantage if True"""
    normalize_return: bool = False
    """ Normalize the returns if True"""
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
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
    """ Logging steps """
    eval_steps: int = 50
    """ Evaluate the policy each «eval_steps» training steps"""
    num_eval_ep: int = 5
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
        normalize_reward=False,
        device="cpu",
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.normalize_reward = normalize_reward
        self.device = device
        self.episodes = [None] * buffer_size
        self.pos = 0

    def add(self, episode):
        for key, values in episode.items():
            episode[key] = torch.from_numpy(np.stack(values)).float().to(self.device)
        self.episodes[self.pos] = episode
        self.pos += 1

    def get_batch(self):
        self.pos = 0
        lengths = [len(episode["obs"]) for episode in self.episodes]
        max_length = max(lengths)
        obs = torch.zeros((self.buffer_size, max_length, self.num_agents, self.obs_space)).to(self.device)
        avail_actions = torch.zeros((self.buffer_size, max_length, self.num_agents, self.action_space)).to(
            self.device
        )
        actions = torch.zeros((self.buffer_size, max_length, self.num_agents)).to(self.device)
        log_probs = torch.zeros((self.buffer_size, max_length, self.num_agents)).to(self.device)
        reward = torch.zeros((self.buffer_size, max_length)).to(self.device)
        states = torch.zeros((self.buffer_size, max_length, self.state_space)).to(self.device)
        done = torch.ones((self.buffer_size, max_length)).to(self.device)
        mask = torch.zeros(self.buffer_size, max_length, dtype=torch.bool).to(self.device)
        for i in range(self.buffer_size):
            length = lengths[i]
            obs[i, :length] = self.episodes[i]["obs"]
            avail_actions[i, :length] = self.episodes[i]["avail_actions"]
            actions[i, :length] = self.episodes[i]["actions"]
            log_probs[i, :length] = self.episodes[i]["log_prob"]
            reward[i, :length] = self.episodes[i]["reward"]
            states[i, :length] = self.episodes[i]["states"]
            done[i, :length] = self.episodes[i]["done"]
            mask[i, :length] = 1
        if self.normalize_reward:
            reward = (reward - reward[mask].mean()) / (reward[mask].std() + 1e-6)
        self.episodes = [None] * self.buffer_size
        return (
            obs.float(),
            actions.long(),
            log_probs.float(),
            reward.float(),
            states.float(),
            avail_actions.bool(),
            done.float(),
            mask,
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


def environment(env_type, env_name, env_family, agent_ids, kwargs):
    if env_type == "pz":
        env = PettingZooWrapper(family=env_family, env_name=env_name, agent_ids=agent_ids, **kwargs)
    elif env_type == "smaclite":
        env = SMACliteWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    elif env_type == "lbf":
        env = LBFWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)

    return env


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.tensor(norms), d)
    return total_norm_d


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, env):
        self.env = env

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.env)

    def __setstate__(self, env):
        import pickle

        self.env = pickle.loads(env)


def env_worker(conn, env_serialized):
    env = env_serialized.env
    while True:
        task, content = conn.recv()
        if task == "reset":
            obs, _ = env.reset(seed=content)
            avail_actions = env.get_avail_actions()
            state = env.get_state()
            content = {"obs": obs, "avail_actions": avail_actions, "state": state}
            conn.send(content)
        elif task == "get_env_info":
            content = {
                "obs_size": env.get_obs_size(),
                "action_size": env.get_action_size(),
                "n_agents": env.n_agents,
                "state_size": env.get_state_size(),
            }
            conn.send(content)
        elif task == "sample":
            actions = env.sample()
            content = {"actions": actions}
            conn.send(content)
        elif task == "step":
            next_obs, reward, done, truncated, infos = env.step(content)
            avail_actions = env.get_avail_actions()
            state = env.get_state()
            content = {
                "next_obs": next_obs,
                "reward": reward,
                "done": done,
                "truncated": truncated,
                "infos": infos,
                "avail_actions": avail_actions,
                "next_state": state,
            }
            conn.send(content)
        elif task == "close":
            env.close()
            conn.close()
            break


if __name__ == "__main__":
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
    ## import the environment
    kwargs = {}  # {"render_mode":'human',"shared_reward":False}
    conns = [Pipe() for _ in range(args.n_episodes)]
    mappo_conns, env_conns = zip(*conns)
    envs = [
        CloudpickleWrapper(
            environment(
                env_type=args.env_type,
                env_name=args.env_name,
                env_family=args.env_family,
                agent_ids=args.agent_ids,
                kwargs=kwargs,
            )
        )
        for _ in range(args.n_episodes)
    ]
    processes = [Process(target=env_worker, args=(env_conns[i], envs[i])) for i in range(args.n_episodes)]
    for process in processes:
        process.daemon = True
        process.start()
    eval_env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=kwargs,
    )

    ## Initialize the actor, critic and target-critic networks
    actor = Actor(
        input_dim=eval_env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        output_dim=eval_env.get_action_size(),
    ).to(device)
    critic = Critic(
        input_dim=eval_env.get_state_size(),
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
    ).to(device)

    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{args.exp_name}__{time_token}"
    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"MAPPO-lstm-multienvs-{run_name}",
        )
    writer = SummaryWriter(f"{args.work_dir}/MAPPO-lstm-multienvs-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )

    rb = RolloutBuffer(
        buffer_size=args.n_episodes,
        obs_space=eval_env.get_obs_size(),
        state_space=eval_env.get_state_size(),
        action_space=eval_env.get_action_size(),
        num_agents=eval_env.n_agents,
        normalize_reward=args.normalize_reward,
        device=device,
    )
    ep_rewards, ep_lengths, ep_stats = [], [], []
    ac_losses, cr_losses, entropies = [], [], []
    ac_gradients, cr_gradients = [], []
    kl_divs, clipped_ratios = [], []
    step, num_episodes = 0, 0
    while step < args.total_timesteps:
        episodes = [
            {
                "obs": [],
                "actions": [],
                "log_prob": [],
                "reward": [],
                "states": [],
                "done": [],
                "avail_actions": [],
            }
            for _ in range(args.n_episodes)
        ]

        for i, mappo_conn in enumerate(mappo_conns):
            mappo_conn.send(("reset", seed + i))
        contents = [mappo_conn.recv() for mappo_conn in mappo_conns]
        obs = np.stack([content["obs"] for content in contents], axis=0)
        avail_action = np.stack([content["avail_actions"] for content in contents], axis=0)
        state = np.stack([content["state"] for content in contents])
        alive_envs = list(range(args.n_episodes))
        ep_reward, ep_length, ep_stat = (
            [0] * args.n_episodes,
            [0] * args.n_episodes,
            [0] * args.n_episodes,
        )
        h = None
        while len(alive_envs) > 0:
            with torch.no_grad():
                obs = obs.reshape(len(alive_envs) * eval_env.n_agents, -1)
                avail_action = avail_action.reshape(len(alive_envs) * eval_env.n_agents, -1)
                # use the hidden tensor just for live environments
                actions, log_probs, next_h = actor.act(
                    torch.from_numpy(obs).float().to(device),
                    h=h,
                    avail_action=torch.tensor(avail_action).bool().to(device),
                )
                actions, log_probs = actions.cpu().numpy(), log_probs.cpu()
                obs = obs.reshape(len(alive_envs), eval_env.n_agents, -1)
                avail_action = avail_action.reshape(len(alive_envs), eval_env.n_agents, -1)
                actions = actions.reshape(len(alive_envs), eval_env.n_agents)
                log_probs = log_probs.reshape(len(alive_envs), eval_env.n_agents)
                next_h = (
                    next_h[0].reshape(1, len(alive_envs), eval_env.n_agents, -1),
                    next_h[1].reshape(1, len(alive_envs), eval_env.n_agents, -1),
                )
            for i, j in enumerate(alive_envs):
                mappo_conns[j].send(("step", actions[i]))

            contents = [mappo_conns[i].recv() for i in alive_envs]
            next_obs = [content["next_obs"] for content in contents]
            reward = [content["reward"] for content in contents]
            done = [content["done"] for content in contents]
            truncated = [content["truncated"] for content in contents]
            infos = [content.get("infos") for content in contents]
            next_avail_action = [content["avail_actions"] for content in contents]
            next_state = [content["next_state"] for content in contents]
            for i, j in enumerate(alive_envs):
                episodes[j]["obs"].append(obs[i])
                episodes[j]["actions"].append(actions[i])
                episodes[j]["log_prob"].append(log_probs[i])
                episodes[j]["reward"].append(reward[i])
                episodes[j]["states"].append(state[i])
                episodes[j]["done"].append(done[i])
                episodes[j]["avail_actions"].append(avail_action[i])
                ep_reward[j] += reward[i]
                ep_length[j] += 1
            step += len(alive_envs)
            obs, state, avail_action = [], [], []
            h_i = []
            for i, j in enumerate(alive_envs[:]):
                if done[i] or truncated[i]:
                    alive_envs.remove(j)
                    rb.add(episodes[j])
                    episodes[j] = dict()
                    if args.env_type == "smaclite":
                        ep_stat[j] = infos[i]
                else:
                    h_i.append(i)
                    obs.append(next_obs[i])
                    avail_action.append(next_avail_action[i])
                    state.append(next_state[i])
            if obs != []:
                obs = np.stack(obs, axis=0)
                avail_action = np.stack(avail_action, axis=0)
                state = np.stack(state, axis=0)
                h = (next_h[0][:, h_i].flatten(1, 2), next_h[1][:, h_i].flatten(1, 2))

        num_episodes += args.n_episodes
        ep_rewards.extend(ep_reward)
        ep_lengths.extend(ep_length)
        ep_stats.extend(ep_stat)

        ## Collate episodes in buffer into single batch
        (
            b_obs,
            b_actions,
            b_log_probs,
            b_reward,
            b_states,
            b_avail_actions,
            b_done,
            b_mask,
        ) = rb.get_batch()

        # Compute the advantage
        return_lambda = torch.zeros_like(b_reward).float().to(device)
        advantages = torch.zeros_like(b_reward).float().to(device)
        with torch.no_grad():
            for ep_idx in range(return_lambda.size(0)):
                next_value = critic(x=b_states[ep_idx])
                next_value[~b_mask[ep_idx]] = 0
                ep_len = b_mask[ep_idx].sum()
                next_value = torch.cat((next_value, torch.zeros((1,), device=device)))
                last_return_lambda = 0
                for t in reversed(range(ep_len)):
                    return_lambda[ep_idx, t] = last_return_lambda = b_reward[ep_idx, t] + args.gamma * (
                        args.td_lambda * last_return_lambda + (1 - args.td_lambda) * next_value[t + 1]
                    )
                    advantages[ep_idx, t] = return_lambda[ep_idx, t] - next_value[t]
        if args.normalize_advantage:
            advantages = (advantages - advantages[b_mask].mean()) / (advantages[b_mask].std() + 1e-8)
        if args.normalize_return:
            return_lambda = (return_lambda - return_lambda[b_mask].mean()) / (
                return_lambda[b_mask].std() + 1e-8
            )
        # training loop
        b_obs = b_obs.permute(0, 2, 1, 3).flatten(0, 1)
        b_actions = b_actions.permute(0, 2, 1).flatten(0, 1)
        b_log_probs = b_log_probs.permute(0, 2, 1).flatten(0, 1)
        b_avail_actions = b_avail_actions.permute(0, 2, 1, 3).flatten(0, 1)
        advantages = advantages.unsqueeze(-1).expand(-1, -1, eval_env.n_agents).permute(0, 2, 1).flatten(0, 1)
        b_mask_ = b_mask.unsqueeze(-1).expand(-1, -1, eval_env.n_agents).permute(0, 2, 1).flatten(0, 1)
        for _ in range(args.epochs):
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
                ## Compute PG the loss
                pg_loss1 = advantages[:, start:end] * ratio
                pg_loss2 = advantages[:, start:end] * torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip)
                pg_loss = torch.min(pg_loss1[b_mask_[:, start:end]], pg_loss2[b_mask_[:, start:end]]).mean()
                # Compute entropy bonus
                entropy_loss = entropy_loss[b_mask_[:, start:end]].mean()
                actor_loss = -pg_loss - args.entropy_coef * entropy_loss
                entropies.append(entropy_loss.item())
                ac_losses.append(pg_loss.item())
                # Compute the value loss
                current_values = critic(x=b_states[:, start:end])
                critic_loss = F.mse_loss(
                    current_values[b_mask[:, start:end]],
                    return_lambda[:, start:end][b_mask[:, start:end]],
                )
                cr_losses.append(critic_loss.item())
                # update networks
                critic_optimizer.zero_grad()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                ac_gradient = norm_d([p.grad for p in actor.parameters()], 2)
                cr_gradient = norm_d([p.grad for p in critic.parameters()], 2)
                ac_gradients.append(ac_gradient.item())
                cr_gradients.append(cr_gradient.item())
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.clip_gradients)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.clip_gradients)
                actor_optimizer.step()
                critic_optimizer.step()
                h = (h[0].detach(), h[1].detach())
                # track kl distance
                with torch.no_grad():
                    kl_div = ((ratio[b_mask_[:, start:end]] - 1) - log_ratio[b_mask_[:, start:end]]).mean()
                    kl_divs.append(kl_div.item())
                    clipped_ratio = (
                        ((ratio[b_mask_[:, start:end]] - 1.0).abs() > args.ppo_clip).float().mean()
                    )
                    clipped_ratios.append(clipped_ratio.item())

        ## logging
        if len(ep_rewards) > args.log_every:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "rollout/battle_won",
                    np.mean([info["battle_won"] for info in ep_stats]),
                    step,
                )
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
            ep_rewards, ep_lengths, ep_stats = [], [], []

        if (num_episodes / args.n_episodes) % args.eval_steps == 0 or step >= args.total_timesteps - 1:
            eval_obs, _ = eval_env.reset()
            eval_ep, current_reward, current_ep_length = 0, 0, 0
            eval_ep_reward, eval_ep_length, eval_ep_stats = [], [], []
            h_eval = None
            while eval_ep < args.num_eval_ep:
                with torch.no_grad():
                    logits, h_eval = actor.logits(
                        torch.from_numpy(eval_obs).float().to(device),
                        h=h_eval,
                        avail_action=torch.from_numpy(eval_env.get_avail_actions()).bool().to(device),
                    )
                    actions = logits.argmax(-1).squeeze(1)
                next_obs_, reward, done, truncated, infos = eval_env.step(actions.cpu().numpy())
                current_reward += reward
                current_ep_length += 1
                eval_obs = next_obs_
                if done or truncated:
                    eval_obs, _ = eval_env.reset()
                    eval_ep_reward.append(current_reward)
                    eval_ep_length.append(current_ep_length)
                    eval_ep_stats.append(infos)
                    current_reward, current_ep_length = 0, 0
                    eval_ep += 1
                    h_eval = None
            writer.add_scalar("eval/ep_reward", np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/std_ep_reward", np.std(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length", np.mean(eval_ep_length), step)
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "eval/battle_won",
                    np.mean([info["battle_won"] for info in eval_ep_stats]),
                    step,
                )
    if args.save_model:
        # Save the weights
        actor_model_path = f"{args.work_dir}/MAPPO-lstm-multienvs-{run_name}/actor.pt"
        torch.save(actor.state_dict(), actor_model_path)
        critic_model_path = f"{args.work_dir}/MAPPO-lstm-multienvs-{run_name}/critic.pt"
        torch.save(critic.state_dict(), critic_model_path)

        # Save the args
        import json
        from dataclasses import asdict

        mappo_args_path = f"{args.work_dir}/MAPPO-lstm-multienvs-{run_name}/args.json"
        with open(mappo_args_path, "w") as f:
            json.dump(asdict(args), f, indent=2)
    writer.close()
    if args.use_wnb:
        wandb.finish()
    eval_env.close()
    for conn in mappo_conns:
        conn.send(("close", None))
    for process in processes:
        process.join()
