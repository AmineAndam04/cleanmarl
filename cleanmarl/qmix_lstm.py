import copy
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
import tyro
import random
from env.pettingzoo_wrapper import PettingZooWrapper
from env.smaclite_wrapper import SMACliteWrapper
from env.lbf import LBFWrapper
import torch.nn.functional as F
import datetime
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    env_type: str = "smaclite"  # "pz"
    """ pz(for Pettingzoo), smaclite (for SMAClite), lbf (for LBF) ... """
    env_name: str = "3m"  # "simple_spread_v3" #"pursuit_v4"
    """ Name of the environment """
    env_family: str = "mpe"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    buffer_size: int = 5000
    """ The number of episodes in the replay buffer"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    train_freq: int = 1
    """ Train the network each «train_freq» step in the environment"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate: float = 0.0008
    """ Learning rate"""
    batch_size: int = 3
    """ Batch size"""
    start_e: float = 1
    """ The starting value of epsilon, for exploration"""
    end_e: float = 0.05
    """ The end value of epsilon, for exploration"""
    exploration_fraction: float = 0.05
    """ The fraction of «total-timesteps» it takes from to go from start_e to  end_e"""
    hidden_dim: int = 32
    """ Hidden dimension"""
    hyper_dim: int = 32
    """ Hidden dimension of hyper-network"""
    num_layers: int = 1
    """ Number of layers"""
    target_network_update_freq: int = 1
    """ Update the target network each target_network_update_freq» step in the environment"""
    polyak: float = 0.005
    """ Polyak coefficient when using polyak averaging for target network update"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    tbptt: int = 10
    """Chunck size for Truncated Backpropagation Through Time tbptt"""
    log_every: int = 10
    """ Log rollout stats every <log_every> episode """
    eval_steps: int = 50
    """ Evaluate the policy each «eval_steps» episode"""
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""
    save_model: bool = True
    """ If True, save the weights of the agents and hyperparameters"""
    device: str = "cpu"
    """ Device (cpu, cuda, mps)"""
    seed: int = 1
    """ Random seed"""


class Qnetwrok(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc2 = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x, h=None, avail_action=None):
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
            x = x.masked_fill(~avail_action, float("-inf"))
        return x, h


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
        W1 = W1.reshape(-1, self.n_agents, self.hidden_dim)
        b1 = self.hypernet_bias_1(s)
        b1 = b1.reshape(-1, 1, self.hidden_dim)
        Q = nn.functional.elu(torch.bmm(Q, W1) + b1)

        W2 = torch.abs(self.hypernet_weight_2(s))
        W2 = W2.reshape(-1, self.hidden_dim, 1)
        b2 = self.hypernet_bias_2(s)
        b2 = b2.reshape(-1, 1, 1)
        Q_tot = torch.bmm(Q, W2) + b2
        return Q_tot


class Batch(NamedTuple):
    batch_obs: torch.Tensor
    batch_action: torch.Tensor
    batch_reward: torch.Tensor
    batch_next_obs: torch.Tensor
    batch_states: torch.Tensor
    batch_next_states: torch.Tensor
    batch_avail_action: torch.Tensor
    batch_done: torch.Tensor
    batch_mask: torch.Tensor


class ReplayBuffer:
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
        self.size = 0

    def store(self, episode):
        for key, values in episode.items():
            episode[key] = torch.from_numpy(np.stack(values)).float().to(self.device)
        self.episodes[self.pos] = episode  # (obs,action,reward,done,next_obs,mask)
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = [self.episodes[i] for i in indices]
        lengths = [len(episode["obs"]) - 1 for episode in batch]
        max_length = max(lengths)
        obs = torch.zeros((batch_size, max_length, self.num_agents, self.obs_space)).to(self.device)
        avail_actions = torch.zeros(
            (batch_size, max_length, self.num_agents, self.action_space)
        ).to(self.device)
        actions = torch.zeros((batch_size, max_length, self.num_agents)).to(self.device)
        reward = torch.zeros((batch_size, max_length)).to(self.device)
        next_obs = torch.zeros((batch_size, max_length, self.num_agents, self.obs_space)).to(
            self.device
        )
        states = torch.zeros((batch_size, max_length, self.state_space)).to(self.device)
        next_states = torch.zeros((batch_size, max_length, self.state_space)).to(self.device)
        done = torch.ones((batch_size, max_length)).to(self.device)
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool).to(self.device)

        for i in range(batch_size):
            length = lengths[i]
            obs[i, :length] = batch[i]["obs"][:-1]
            avail_actions[i, :length] = batch[i]["avail_actions"]
            actions[i, :length] = batch[i]["actions"]
            reward[i, :length] = batch[i]["reward"]
            next_obs[i, :length] = batch[i]["obs"][1:]
            states[i, :length] = batch[i]["states"][:-1]
            next_states[i, :length] = batch[i]["states"][1:]
            done[i, :length] = batch[i]["done"]
            mask[i, :length] = 1

        if self.normalize_reward:
            mu = reward[mask].mean()
            std = reward[mask].std()
            reward = (reward - mu) / (std + 1e-6)
        return Batch(
            batch_obs=obs.float().permute(0, 2, 1, 3),
            batch_action=actions.long().permute(0, 2, 1),
            batch_reward=reward.float(),
            batch_next_obs=next_obs.float().permute(0, 2, 1, 3),
            batch_states=states.float(),
            batch_next_states=next_states.float(),
            batch_avail_action=avail_actions.bool().permute(0, 2, 1, 3),
            batch_done=done.float(),
            batch_mask=mask,
        )


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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


def soft_update(target_net, utility_net, polyak):
    for target_param, param in zip(target_net.parameters(), utility_net.parameters()):
        target_param.data.copy_(polyak * param.data + (1.0 - polyak) * target_param.data)


def get_mini_batches(batch, t, minibatch_size):
    return (
        batch.batch_obs[:, :, t : t + minibatch_size].flatten(
            0, 1
        ),  # I flatten so the batch_size = num_envs*num_agents as I cannot batch over the agents as with MLPs
        batch.batch_action[:, :, t : t + minibatch_size].flatten(0, 1),
        batch.batch_reward[:, t : t + minibatch_size],
        batch.batch_next_obs[:, :, t : t + minibatch_size].flatten(0, 1),
        batch.batch_states[:, t : t + minibatch_size],
        batch.batch_next_states[:, t : t + minibatch_size],
        batch.batch_avail_action[:, :, t : t + minibatch_size].flatten(0, 1),
        batch.batch_done[:, t : t + minibatch_size],
        batch.batch_mask[:, t : t + minibatch_size],
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Set the seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    ## import the environment
    kwargs = {}  # {"render_mode":'human',"shared_reward":False}
    env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=kwargs,
    )
    eval_env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=kwargs,
    )
    ## initialize the utility and target networks
    utility_network = Qnetwrok(
        input_dim=env.get_obs_size(),
        hidden_dim=args.hidden_dim,
        output_dim=env.get_action_size(),
    ).to(device)
    target_network = copy.deepcopy(utility_network).to(device)
    mixer = MixingNetwork(
        n_agents=env.n_agents, s_dim=env.get_state_size(), hidden_dim=args.hyper_dim
    ).to(device)
    target_mixer = copy.deepcopy(mixer).to(device)

    ## initialize the optimizer
    optimizer = getattr(optim, args.optimizer)
    optimizer = optimizer(
        list(utility_network.parameters()) + list(mixer.parameters()),
        lr=args.learning_rate,
    )

    ## initialize a shared replay buffer
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space=env.get_obs_size(),
        state_space=env.get_state_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        normalize_reward=args.normalize_reward,
        device=device,
    )

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"QMIX-lstm-{run_name}",
        )
    writer = SummaryWriter(f"runs/QMIX-lstm-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    num_episode = 0
    num_updates = 0
    step = 0
    while step < args.total_timesteps:
        episode = {
            "obs": [],
            "actions": [],
            "reward": [],
            "states": [],
            "done": [],
            "avail_actions": [],
        }
        obs, _ = env.reset(seed=seed)
        avail_action = env.get_avail_actions()
        state = env.get_state()
        ep_reward, ep_length = 0, 0
        done, truncated = False, False
        h = None
        while not done and not truncated:
            epsilon = linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * args.total_timesteps,
                step,
            )
            with torch.no_grad():
                q_values, h = utility_network(
                    torch.from_numpy(obs).float().to(device),
                    h=h,
                    avail_action=torch.from_numpy(avail_action).bool().to(device),
                )
                q_values = q_values.squeeze(1)
            if random.random() < epsilon:
                actions = env.sample()
            else:
                actions = torch.argmax(q_values, dim=-1).cpu().numpy()
            next_obs, reward, done, truncated, infos = env.step(actions)
            avail_action = (
                env.get_avail_actions()
            )  # Get the mask of 'next_obs' and store it in the replay, we need it for the bellman loss
            next_state = env.get_state()

            ep_reward += reward
            ep_length += 1
            step += 1
            episode["obs"].append(obs)
            episode["actions"].append(actions)
            episode["reward"].append(reward)
            episode["done"].append(done)
            episode["avail_actions"].append(avail_action)
            episode["states"].append(state)
            obs = next_obs
            state = next_state
        episode["obs"].append(obs)
        episode["states"].append(state)
        rb.store(episode)
        num_episode += 1
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)
        if args.env_type == "smaclite":
            ep_stats.append(infos)  ## Add battle won for smaclite

        if num_episode > args.batch_size:
            if num_episode % args.train_freq == 0:
                batch = rb.sample(args.batch_size)
                losses = []
                gradients = []
                h_target = None
                h_utility = None
                for t in range(0, batch.batch_obs.size(1), args.tbptt):
                    (
                        mb_obs,
                        mb_action,
                        mb_reward,
                        mb_next_obs,
                        mb_states,
                        mb_next_states,
                        mb_avail_action,
                        mb_done,
                        mb_mask,
                    ) = get_mini_batches(batch, t, args.tbptt)
                    with torch.no_grad():
                        q_next, h_target = target_network(
                            mb_next_obs,
                            h=h_target,
                            avail_action=mb_avail_action,
                        )
                        q_next_max, _ = q_next.max(dim=-1)
                        q_next_max = q_next_max.reshape(args.batch_size, -1, env.n_agents)
                        q_tot_target = target_mixer(Q=q_next_max, s=mb_next_states)
                        q_tot_target = q_tot_target.reshape(args.batch_size, -1)
                        targets = mb_reward + args.gamma * (1 - mb_done) * q_tot_target

                    q_values, h_utility = utility_network(mb_obs, h=h_utility)
                    q_values = torch.gather(q_values, dim=-1, index=mb_action.unsqueeze(-1))
                    q_values = q_values.reshape(args.batch_size, -1, env.n_agents)
                    q_tot = mixer(Q=q_values, s=mb_states)
                    q_tot = q_tot.reshape(args.batch_size, -1)
                    loss = F.mse_loss(targets[mb_mask], q_tot[mb_mask])
                    optimizer.zero_grad()
                    loss.backward()
                    loss_gradients = norm_d(
                        [
                            p.grad
                            for p in list(utility_network.parameters()) + list(mixer.parameters())
                        ],
                        2,
                    )
                    gradients.append(loss_gradients)
                    losses.append(loss.item())
                    if args.clip_gradients > 0:
                        torch.nn.utils.clip_grad_norm_(
                            list(utility_network.parameters()) + list(mixer.parameters()),
                            args.clip_gradients,
                        )
                    optimizer.step()
                    num_updates += 1
                    h_utility = (h_utility[0].detach(), h_utility[1].detach())

                writer.add_scalar("train/loss", np.mean(losses), step)
                writer.add_scalar("train/grads", np.mean(gradients), step)
                writer.add_scalar("train/num_updates", num_updates, step)

            if num_episode % args.target_network_update_freq == 0:
                soft_update(
                    target_net=target_network,
                    utility_net=utility_network,
                    polyak=args.polyak,
                )
                soft_update(target_net=target_mixer, utility_net=mixer, polyak=args.polyak)
        if num_episode % args.log_every == 0:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            writer.add_scalar("rollout/epsilon", epsilon, step)
            writer.add_scalar("rollout/num_episodes", num_episode, step)
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "rollout/battle_won",
                    np.mean([info["battle_won"] for info in ep_stats]),
                    step,
                )
            ep_rewards = []
            ep_lengths = []
            ep_stats = []

        if num_episode % args.eval_steps == 0:
            eval_obs, _ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            h_eval = None
            while eval_ep < args.num_eval_ep:
                q_values, h_eval = utility_network(
                    torch.from_numpy(eval_obs).float().to(device),
                    h=h_eval,
                    avail_action=torch.tensor(eval_env.get_avail_actions(), dtype=torch.bool).to(
                        device
                    ),
                )
                actions = torch.argmax(q_values, dim=-1).reshape(eval_env.n_agents).cpu().numpy()
                next_obs_, reward, done, truncated, infos = eval_env.step(actions)
                current_reward += reward
                current_ep_length += 1
                eval_obs = next_obs_
                if done or truncated:
                    eval_obs, _ = eval_env.reset()
                    h_eval = None
                    eval_ep_reward.append(current_reward)
                    eval_ep_length.append(current_ep_length)
                    eval_ep_stats.append(infos)
                    current_reward = 0
                    current_ep_length = 0
                    eval_ep += 1
            writer.add_scalar("eval/ep_reward", np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/std_ep_reward", np.std(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length", np.mean(eval_ep_length), step)
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "eval/battle_won",
                    np.mean(np.mean([info["battle_won"] for info in eval_ep_stats])),
                    step,
                )

    if args.save_model:
        # Save the weights
        qmix_model_path = f"runs/QMIX-lstm-{run_name}/agent.pt"
        torch.save(utility_network.state_dict(), qmix_model_path)
        mixer_model_path = f"runs/QMIX-lstm-{run_name}/mixer.pt"
        torch.save(mixer.state_dict(), mixer_model_path)

        # Save the args
        import json
        from dataclasses import asdict

        qmix_args_path = f"runs/QMIX-lstm-{run_name}/args.json"
        with open(qmix_args_path, "w") as f:
            json.dump(asdict(args), f, indent=2)
    writer.close()
    if args.use_wnb:
        wandb.finish()
    env.close()
    eval_env.close()
