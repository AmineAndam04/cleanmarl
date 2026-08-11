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
from env.pettingzoo_wrapper import PettingZooWrapper
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    # Environment
    env_type: str = "pz"
    """ Pettingzoo, ... """
    env_name: str = "multiwalker_v9"
    """ Name of the environment """
    env_family: str = "sisl"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
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
    """ Batch size"""
    train_freq: int = 1
    """ Train the network each «train_freq» step in the environment"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.00001
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.00001
    """ Learning rate for the critic"""
    gamma: float = 0.99
    """ Discount factor"""
    normalize_reward: bool = True
    """ Normalize the rewards if True"""
    clip_gradients: float = 0.5
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


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for _ in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))

    def act(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Qnetwrok(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for _ in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, 1)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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
            episode  # {"obs": [],"actions":[],"reward":[],"states":[],"done":[]}
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
            done.float(),
            mask,
        )


def environment(env_type, env_name, env_family, agent_ids, kwargs):
    if env_type == "pz":
        env = PettingZooWrapper(
            family=env_family, env_name=env_name, agent_ids=agent_ids, **kwargs
        )
    return env


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
    act_low = torch.from_numpy(env.act_low)
    act_high = torch.from_numpy(env.act_high)
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

    mixer = MixingNetwork(
        n_agents=env.n_agents, s_dim=env.get_state_size(), hidden_dim=args.hyper_dim
    ).to(device)
    target_mixer = copy.deepcopy(mixer).to(device)

    ## initialize the optimizer
    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(
        list(critic.parameters()) + list(mixer.parameters()),
        lr=args.learning_rate_critic,
    )

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
    writer = SummaryWriter(f"{args.work_dir}/FACMAC-continuous-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space=env.get_obs_size(),
        state_space=env.get_state_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        normalize_reward=args.normalize_reward,
        device=device,
    )
    ep_rewards, ep_lengths = [], []
    cr_losses, cr_gradients, ac_losses, ac_gradients = [], [], [], []
    step, num_episodes = 0, 0
    while step < args.total_timesteps:
        episode = {
            "obs": [],
            "actions": [],
            "reward": [],
            "states": [],
            "done": [],
        }
        obs, _ = env.reset()
        ep_reward, ep_length = 0, 0
        done, truncated = False, False
        while not done and not truncated:
            state = env.get_state()
            with torch.no_grad():
                actions = actor.act(torch.from_numpy(obs).float().to(device))
                noise = 0.05 * torch.randn_like(actions)
                actions = torch.clamp(actions + noise, act_low, act_high).cpu().numpy()
            next_obs, reward, done, truncated, infos = env.step(actions)
            ep_reward += reward
            ep_length += 1
            step += 1
            episode["obs"].append(obs)
            episode["actions"].append(actions)
            episode["reward"].append(reward)
            episode["done"].append(done)
            episode["states"].append(state)
            obs = next_obs
        rb.store(episode)

        num_episodes += 1
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)

        if num_episodes > args.batch_size:
            if num_episodes % args.train_freq == 0:
                (
                    batch_obs,
                    batch_action,
                    batch_reward,
                    batch_states,
                    batch_done,
                    batch_mask,
                ) = rb.sample(args.batch_size)
                ## train the critic
                critic_loss = 0
                for t in range(batch_obs.size(1)):
                    with torch.no_grad():
                        if t == (batch_obs.size(1) - 1):
                            targets = batch_reward[:, t]
                        else:
                            actions_from_target_actor = target_actor.act(
                                batch_obs[:, t + 1]
                            )
                            actions_from_target_actor = torch.clamp(
                                actions_from_target_actor, act_low, act_high
                            )
                            qvals_from_taget_utility = target_critic(
                                torch.cat(
                                    (batch_obs[:, t + 1], actions_from_target_actor),
                                    dim=-1,
                                )
                            ).squeeze()
                            q_tot_from_target_mixer = target_mixer(
                                Q=qvals_from_taget_utility, s=batch_states[:, t + 1]
                            ).squeeze()
                            q_tot_from_target_mixer = torch.nan_to_num(
                                q_tot_from_target_mixer, nan=0.0
                            )
                            targets = (
                                batch_reward[:, t]
                                + args.gamma
                                * (1 - batch_done[:, t])
                                * q_tot_from_target_mixer
                            )
                    q_values = critic(
                        torch.cat((batch_obs[:, t], batch_action[:, t]), dim=-1)
                    ).squeeze()
                    q_tot = mixer(Q=q_values, s=batch_states[:, t]).squeeze()
                    critic_loss += F.mse_loss(
                        targets[batch_mask[:, t]], q_tot[batch_mask[:, t]]
                    ) * (batch_mask[:, t].sum())
                critic_loss /= batch_mask.sum()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_gradient = norm_d([p.grad for p in critic.parameters()], 2)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(
                        critic.parameters(), max_norm=args.clip_gradients
                    )
                    torch.nn.utils.clip_grad_norm_(
                        mixer.parameters(), max_norm=args.clip_gradients
                    )
                critic_optimizer.step()
                cr_losses.append(critic_loss.item())
                cr_gradients.append(critic_gradient.item())

                perm = torch.randperm(batch_obs.size(1) - 1)
                actor_loss = 0
                for t in perm:
                    actions = actor.act(batch_obs[:, t])
                    actions = torch.clamp(actions, act_low, act_high)
                    q_values = critic(
                        torch.cat((batch_obs[:, t], actions), dim=-1)
                    ).squeeze()
                    q_tot = mixer(Q=q_values, s=batch_states[:, t]).squeeze()
                    actor_loss -= q_tot[batch_mask[:, t]].sum()
                actor_loss /= batch_mask.sum()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_gradient = norm_d([p.grad for p in actor.parameters()], 2)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), max_norm=args.clip_gradients
                    )
                actor_optimizer.step()
                ac_losses.append(actor_loss.item())
                ac_gradients.append(actor_gradient.item())

            if num_episodes % args.target_network_update_freq == 0:
                soft_update(
                    target_net=target_actor, utility_net=actor, polyak=args.polyak
                )
                soft_update(
                    target_net=target_critic, utility_net=critic, polyak=args.polyak
                )
                soft_update(
                    target_net=target_mixer, utility_net=mixer, polyak=args.polyak
                )

        if num_episodes % args.log_every == 0:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            if len(ac_losses) > 0:
                writer.add_scalar("train/critic_loss", np.mean(cr_losses), step)
                writer.add_scalar("train/critic_gradients", np.mean(cr_gradients), step)
                writer.add_scalar("train/actor_loss", np.mean(ac_losses), step)
                writer.add_scalar("train/actor_gradients", np.mean(ac_gradients), step)
                cr_losses, cr_gradients, ac_losses, ac_gradients = [], [], [], []
            ep_rewards, ep_lengths = [], []

        if num_episodes % args.eval_steps == 0 or step >= args.total_timesteps - 1:
            eval_obs, _ = eval_env.reset()
            eval_ep_reward, eval_ep_length = [], []
            eval_ep, current_reward, current_ep_length = 0, 0, 0
            while eval_ep < args.num_eval_ep:
                with torch.no_grad():
                    eval_actions = actor.act(
                        torch.from_numpy(eval_obs).float().to(device)
                    )
                    eval_actions = torch.clamp(eval_actions, act_low, act_high)
                next_obs_, reward, done, truncated, infos = eval_env.step(
                    eval_actions.cpu().numpy()
                )
                current_reward += reward
                current_ep_length += 1
                eval_obs = next_obs_
                if done or truncated:
                    eval_obs, _ = eval_env.reset()
                    eval_ep_reward.append(current_reward)
                    eval_ep_length.append(current_ep_length)
                    current_reward, current_ep_length = 0, 0
                    eval_ep += 1
            writer.add_scalar("eval/ep_reward", np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/std_ep_reward", np.std(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length", np.mean(eval_ep_length), step)
    if args.save_model:
        # Save the weights
        actor_model_path = f"{args.work_dir}/FACMAC-continuous-{run_name}/agent.pt"
        torch.save(actor.state_dict(), actor_model_path)
        critic_model_path = f"{args.work_dir}/FACMAC-continuous-{run_name}/critic.pt"
        torch.save(critic.state_dict(), critic_model_path)
        mixer_model_path = f"{args.work_dir}/FACMAC-continuous-{run_name}/mixer.pt"
        torch.save(mixer.state_dict(), mixer_model_path)

        # Save the args
        import json
        from dataclasses import asdict

        facmac_args_path = f"{args.work_dir}/FACMAC-continuous-{run_name}/args.json"
        with open(facmac_args_path, "w") as f:
            json.dump(asdict(args), f, indent=2)

    writer.close()
    if args.use_wnb:
        wandb.finish()
    env.close()
    eval_env.close()
