import copy
from multiprocessing import Pipe, Process
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
    """ Pettingzoo, SMAClite ... """
    env_name: str = "3m"  # "simple_spread_v3" #"pursuit_v4"
    """ Name of the environment """
    env_family: str = "mpe"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    num_envs: int = 4
    """ Number of parallel environments"""
    gamma: float = 0.99
    """ Discount factor"""
    buffer_size: int = 5000
    """ The number of episodes in the replay buffer"""
    batch_size: int = 10
    """ Batch size"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    actor_hidden_dim: int = 32
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 128
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0003
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0003
    """ Learning rate for the critic"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    target_network_update_freq: int = 1
    """ Update the target network each target_network_update_freq» step in the environment"""
    polyak: float = 0.01
    """ Polyak coefficient when using polyak averaging for target network update"""
    epochs: int = 4
    """ In this case, by train_freq we main number of training epochs after collecting num_envs episodes, one epoch = sample from the replay buffer"""
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    tbptt: int = 10
    """Chunck size for Truncated Backpropagation Through Time tbptt"""
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
    device: str = "cpu"
    """ Device (cpu, cuda, mps)"""
    seed: int = 1
    """ Random seed"""


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def act(self, x, h, avail_action=None, hard=False):
        x, h = self.logits(x, h, avail_action)
        actions = F.gumbel_softmax(logits=x, hard=hard)
        return actions, h

    def logits(self, x, h, avail_action):
        x = self.fc1(x)
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        h = self.gru(x, h)
        x = self.fc2(h)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, -1e9)
        return x, h


class Critic(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_layer, output_dim, num_agents
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, 1)))

    def forward(self, state, actions, grad_processing=False, batch_action=None):
        x = self.maddpg_inputs(state, actions, grad_processing, batch_action)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()

    def maddpg_inputs(self, state, actions, grad_processing, batch_action):
        maddpg_inputs = torch.zeros(
            (state.size(0), self.num_agents, self.input_dim)
        ).to(state.device)
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
        self.episodes[self.pos] = (
            episode  # {"obs": [],"actions":[],"reward":[],"states":[],"done":[],"avail_actions":[]}
        )
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = [self.episodes[i] for i in indices]
        lengths = [len(episode["obs"]) for episode in batch]
        max_length = max(lengths)
        obs = torch.zeros((batch_size, max_length, self.num_agents, self.obs_space)).to(
            self.device
        )
        avail_actions = torch.zeros(
            (batch_size, max_length, self.num_agents, self.action_space)
        ).to(self.device)
        actions = torch.zeros(
            (batch_size, max_length, self.num_agents, self.action_space)
        ).to(self.device)
        reward = torch.zeros((batch_size, max_length)).to(self.device)
        states = torch.zeros((batch_size, max_length, self.state_space)).to(self.device)
        done = torch.ones((batch_size, max_length)).to(self.device)
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool).to(self.device)

        for i in range(batch_size):
            length = lengths[i]
            obs[i, :length] = batch[i]["obs"]
            avail_actions[i, :length] = batch[i]["avail_actions"]
            actions[i, :length] = batch[i]["actions"]
            reward[i, :length] = batch[i]["reward"]
            states[i, :length] = batch[i]["states"]
            done[i, :length] = batch[i]["done"]
            mask[i, :length] = 1

        if self.normalize_reward:
            mu = torch.mean(reward[mask])
            std = torch.std(reward[mask])
            reward[mask.bool()] = (reward[mask] - mu) / (std + 1e-6)

        return (
            obs.float(),
            actions.float(),
            reward.float(),
            states.float(),
            avail_actions.bool(),
            done.float(),
            mask,
        )


def environment(env_type, env_name, env_family, agent_ids, kwargs):
    if env_type == "pz":
        env = PettingZooWrapper(
            family=env_family, env_name=env_name, agent_ids=agent_ids, **kwargs
        )
    elif env_type == "smaclite":
        env = SMACliteWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    elif env_type == "lbf":
        env = LBFWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)

    return env


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
            obs, _ = env.reset(seed=random.randint(0, 100000))
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


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.tensor(norms), d)
    return total_norm_d


def soft_update(target_net, utility_net, polyak):
    for target_param, param in zip(target_net.parameters(), utility_net.parameters()):
        target_param.data.copy_(
            polyak * param.data + (1.0 - polyak) * target_param.data
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    ## import the environment
    kwargs = {}  # {"render_mode":'human',"shared_reward":False}
    ## Create the pipes to communicate between the main process (maddpg algorithm) and child processes (envs)
    conns = [Pipe() for _ in range(args.num_envs)]
    maddpg_conns, env_conns = zip(*conns)
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
        for _ in range(args.num_envs)
    ]
    processes = [
        Process(target=env_worker, args=(env_conns[i], envs[i]))
        for i in range(args.num_envs)
    ]
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

    actor = Actor(
        input_dim=eval_env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        output_dim=eval_env.get_action_size(),
    ).to(device)
    target_actor = copy.deepcopy(actor).to(device)

    maddpg_input_dim = (
        eval_env.get_state_size() + eval_env.n_agents * eval_env.get_action_size()
    )
    critic = Critic(
        input_dim=maddpg_input_dim,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        output_dim=eval_env.get_action_size(),
        num_agents=eval_env.n_agents,
    ).to(device)
    target_critic = copy.deepcopy(critic).to(device)

    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"MADDPG-lstm-multienvs-{run_name}",
        )
    writer = SummaryWriter(f"runs/MADDPG-lstm-multienvs-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space=eval_env.get_obs_size(),
        state_space=eval_env.get_state_size(),
        action_space=eval_env.get_action_size(),
        num_agents=eval_env.n_agents,
        normalize_reward=args.normalize_reward,
        device=device,
    )
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    num_episode = 0
    num_updates = 0
    step = 0
    while step < args.total_timesteps:
        episodes = [
            {
                "obs": [],
                "actions": [],
                "reward": [],
                "states": [],
                "done": [],
                "avail_actions": [],
            }
            for _ in range(args.num_envs)
        ]
        for maddpg_conn in maddpg_conns:
            maddpg_conn.send(("reset", None))
        contents = [maddpg_conn.recv() for maddpg_conn in maddpg_conns]
        obs = np.stack([content["obs"] for content in contents], axis=0)
        avail_action = np.stack(
            [content["avail_actions"] for content in contents], axis=0
        )
        state = np.stack([content["state"] for content in contents])
        alive_envs = list(range(args.num_envs))
        ep_reward, ep_length, ep_stat = (
            [0] * args.num_envs,
            [0] * args.num_envs,
            [0] * args.num_envs,
        )
        h = None
        while len(alive_envs) > 0:
            with torch.no_grad():
                obs = obs.reshape(len(alive_envs) * eval_env.n_agents, -1)
                avail_action = avail_action.reshape(
                    len(alive_envs) * eval_env.n_agents, -1
                )
                # use the hidden info just for live environments
                if h is None:
                    alive_h = None
                else:
                    alive_h = h.reshape(args.num_envs, eval_env.n_agents, -1)
                    alive_h = alive_h[alive_envs].reshape(
                        len(alive_envs) * eval_env.n_agents, -1
                    )
                with torch.no_grad():
                    actions, alive_h = actor.act(
                        torch.from_numpy(obs).float().to(device),
                        alive_h,
                        avail_action=torch.from_numpy(avail_action).bool().to(device),
                        hard=True,
                    )  ## These are one hot-vectors
                    actions = actions.reshape(
                        len(alive_envs), eval_env.n_agents, -1
                    ).cpu()
                    actions_to_take = torch.argmax(actions, dim=-1)
                if h is None:
                    h = alive_h
                else:
                    h = h.reshape(args.num_envs, eval_env.n_agents, -1)
                    alive_h = alive_h.reshape(len(alive_envs), eval_env.n_agents, -1)
                    h[alive_envs] = alive_h
                    h = h.reshape(args.num_envs * eval_env.n_agents, -1)
                obs = obs.reshape(len(alive_envs), eval_env.n_agents, -1)
                avail_action = avail_action.reshape(
                    len(alive_envs), eval_env.n_agents, -1
                )

            for i, j in enumerate(alive_envs):
                maddpg_conns[j].send(("step", actions_to_take[i]))

            contents = [maddpg_conns[i].recv() for i in alive_envs]
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
                episodes[j]["reward"].append(reward[i])
                episodes[j]["states"].append(state[i])
                episodes[j]["done"].append(done[i])
                episodes[j]["avail_actions"].append(avail_action[i])
                ep_reward[j] += reward[i]
                ep_length[j] += 1
            step += len(alive_envs)
            obs = []
            state = []
            avail_action = []
            for i, j in enumerate(alive_envs[:]):
                if done[i] or truncated[i]:
                    alive_envs.remove(j)
                    rb.store(episodes[j])
                    episodes[j] = dict()
                    if args.env_type == "smaclite":
                        ep_stat[j] = infos[i]
                else:
                    obs.append(next_obs[i])
                    avail_action.append(next_avail_action[i])
                    state.append(next_state[i])
            if obs:
                obs = np.stack(obs, axis=0)
                avail_action = np.stack(avail_action, axis=0)
                state = np.stack(state, axis=0)

        num_episode += args.num_envs

        ep_rewards.extend(ep_reward)
        ep_lengths.extend(ep_length)
        if args.env_type == "smaclite":
            ep_stats.extend([info["battle_won"] for info in ep_stat])
        ## logging
        if (num_updates // args.epochs) % args.log_every:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            if args.env_type == "smaclite":
                writer.add_scalar("rollout/battle_won", np.mean(ep_stats), step)
            ep_rewards = []
            ep_lengths = []
            ep_stats = []

        # print("num_episode",num_episode)
        if num_episode > args.batch_size:
            # print("I'm in ",num_episode)
            critic_losses = []
            critic_gradients = []
            total_actor_losses = []
            total_actor_gradients = []
            for _ in range(args.epochs):
                (
                    batch_obs,
                    batch_action,
                    batch_reward,
                    batch_states,
                    batch_avail_action,
                    batch_done,
                    batch_mask,
                ) = rb.sample(args.batch_size)
                critic_loss = 0
                h_targ = None
                for t in range(batch_obs.size(1)):
                    with torch.no_grad():
                        if t == batch_obs.size(1) - 1:
                            targets = (
                                batch_reward[:, t]
                                .unsqueeze(-1)
                                .expand(-1, eval_env.n_agents)
                            )
                        else:
                            b_obs_t1 = batch_obs[:, t + 1].reshape(
                                args.batch_size * eval_env.n_agents, -1
                            )
                            b_avail_actions_t1 = batch_avail_action[:, t + 1].reshape(
                                args.batch_size * eval_env.n_agents, -1
                            )
                            actions_from_target_actor, h_targ = target_actor.act(
                                b_obs_t1,
                                h_targ,
                                avail_action=b_avail_actions_t1,
                                hard=True,
                            )
                            actions_from_target_actor = (
                                actions_from_target_actor.reshape(
                                    args.batch_size, eval_env.n_agents, -1
                                )
                            )
                            qvals_from_taget_critic = target_critic(
                                batch_states[:, t + 1], actions_from_target_actor
                            )
                            qvals_from_taget_critic = torch.nan_to_num(
                                qvals_from_taget_critic, nan=0.0
                            )
                            targets = (
                                batch_reward[:, t]
                                .unsqueeze(-1)
                                .expand(-1, eval_env.n_agents)
                                + args.gamma
                                * (
                                    1
                                    - batch_done[:, t]
                                    .unsqueeze(-1)
                                    .expand(-1, eval_env.n_agents)
                                )
                                * qvals_from_taget_critic
                            )
                    q_values = critic(batch_states[:, t], batch_action[:, t])
                    critic_loss += F.mse_loss(
                        targets[batch_mask[:, t]], q_values[batch_mask[:, t]]
                    ) * (batch_mask[:, t].sum())
                critic_loss /= batch_mask.sum()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_gradient = norm_d([p.grad for p in critic.parameters()], 2)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(
                        critic.parameters(), max_norm=args.clip_gradients
                    )
                critic_optimizer.step()
                critic_losses.append(critic_loss.item())
                critic_gradients.append(critic_gradient)

                ## train the actor
                actor_losses = 0
                actor_gradients = []
                h_actor = None
                truncated_actor_loss = None
                actor_loss_denominator = None
                T = None
                for t in range(batch_obs.size(1)):
                    b_obs_t = batch_obs[:, t].reshape(
                        args.batch_size * eval_env.n_agents, -1
                    )
                    b_avail_actions_t = batch_avail_action[:, t].reshape(
                        args.batch_size * eval_env.n_agents, -1
                    )
                    actions, h_actor = actor.act(
                        b_obs_t, h_actor, avail_action=b_avail_actions_t, hard=False
                    )
                    actions = actions.reshape(args.batch_size, eval_env.n_agents, -1)
                    qvals = critic(
                        batch_states[:, t],
                        actions,
                        grad_processing=True,
                        batch_action=batch_action[:, t],
                    )
                    actor_loss = -qvals[batch_mask[:, t]].sum()
                    actor_losses += actor_loss
                    if truncated_actor_loss is None:
                        truncated_actor_loss = actor_loss
                        actor_loss_denominator = batch_mask[:, t].sum()
                        T = 1
                    else:
                        truncated_actor_loss += actor_loss
                        actor_loss_denominator += batch_mask[:, t].sum()
                        T += 1
                    if ((t + 1) % args.tbptt == 0) or (t == (batch_obs.size(1) - 1)):
                        truncated_actor_loss = truncated_actor_loss / (
                            actor_loss_denominator * T
                        )
                        actor_optimizer.zero_grad()
                        truncated_actor_loss.backward()
                        tbptt_actor_gradients = norm_d(
                            [p.grad for p in actor.parameters()], 2
                        )
                        actor_gradients.append(tbptt_actor_gradients)
                        if args.clip_gradients > 0:
                            torch.nn.utils.clip_grad_norm_(
                                actor.parameters(), max_norm=args.clip_gradients
                            )
                        actor_optimizer.step()
                        truncated_actor_loss = None
                        h_actor = h_actor.detach()

                total_actor_losses.append(
                    (actor_losses.item() / batch_mask.sum()).cpu()
                )
                total_actor_gradients.append(np.mean(actor_gradients))
                num_updates += 1

            writer.add_scalar("train/critic_loss", np.mean(critic_losses), step)
            writer.add_scalar("train/actor_loss", np.mean(total_actor_losses), step)
            writer.add_scalar(
                "train/actor_gradients", np.mean(total_actor_gradients), step
            )
            writer.add_scalar("train/critic_gradients", np.mean(critic_gradients), step)
            writer.add_scalar("train/num_updates", num_updates, step)
            if (num_updates // args.epochs) % args.target_network_update_freq:
                soft_update(
                    target_net=target_critic, utility_net=critic, polyak=args.polyak
                )
                soft_update(
                    target_net=target_actor, utility_net=actor, polyak=args.polyak
                )

        if (num_episode / args.num_envs) % args.eval_steps == 0:
            eval_obs, _ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            h_eval = None
            while eval_ep < args.num_eval_ep:
                with torch.no_grad():
                    logits, h_eval = actor.logits(
                        torch.from_numpy(eval_obs).float().to(device),
                        h_eval,
                        avail_action=torch.tensor(eval_env.get_avail_actions())
                        .bool()
                        .to(device),
                    )
                    eval_actions = torch.argmax(logits, dim=-1)
                next_obs_, reward, done, truncated, infos = eval_env.step(eval_actions)
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
                    np.mean([info["battle_won"] for info in eval_ep_stats]),
                    step,
                )

    writer.close()
    if args.use_wnb:
        wandb.finish()
    eval_env.close()
    for conn in maddpg_conns:
        conn.send(("close", None))
    for process in processes:
        process.join()
