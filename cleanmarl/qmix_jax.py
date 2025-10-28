from typing import Tuple, Any
import optax
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
from dataclasses import dataclass
import tyro
import random
from env.pettingzoo_wrapper import PettingZooWrapper
from env.smaclite_wrapper import SMACliteWrapper
from env.lbf import LBFWrapper
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
    buffer_size: int = 5000
    """ The number of episodes in the replay buffer"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    train_freq: int = 1
    """ Train the network each «train_freq» step in the environment"""
    optimizer: str = "adam"
    """ The optimizer"""
    learning_rate: float = 0.0005
    """ Learning rate"""
    batch_size: int = 10
    """ Batch size"""
    start_e: float = 1
    """ The starting value of epsilon, for exploration"""
    end_e: float = 0.025
    """ The end value of epsilon, for exploration"""
    exploration_fraction: float = 0.05
    """ The fraction of «total-timesteps» it takes from to go from start_e to  end_e"""
    hidden_dim: int = 64
    """ Hidden dimension"""
    hyper_dim: int = 64
    """ Hidden dimension of hyper-network"""
    num_layers: int = 1
    """ Number of layers"""
    target_network_update_freq: int = 1
    """ Update the target network each target_network_update_freq» step in the environment"""
    polyak: float = 0.01
    """ Polyak coefficient when using polyak averaging for target network update"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    log_every: int = 10
    """ Log rollout stats every <log_every> episode """
    eval_steps: int = 50
    """ Evaluate the policy each «eval_steps» episode"""
    num_eval_ep: int = 5
    """ Number of evaluation episodes"""
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""
    seed: int = 1
    """ Random seed"""


class Qnetwork(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layer: int,
        output_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        kernel_init = jax.nn.initializers.orthogonal()
        self.layers = nnx.List(
            [
                nnx.Linear(input_dim, hidden_dim, kernel_init=kernel_init, rngs=rngs),
                nnx.relu,
            ]
        )
        for _ in range(num_layer):
            self.layers.append(
                nnx.Linear(hidden_dim, hidden_dim, kernel_init=kernel_init, rngs=rngs)
            )
            self.layers.append(nnx.relu)
        self.layers.append(
            nnx.Linear(hidden_dim, output_dim, kernel_init=kernel_init, rngs=rngs)
        )

    def __call__(self, x: jnp.ndarray, avail_action: jnp.ndarray | None = None):
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = jnp.where(avail_action, x, jnp.finfo(jnp.float32).min)
        return x


class MixingNetwork(nnx.Module):
    def __init__(self, n_agents: int, s_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        kernel_init = jax.nn.initializers.orthogonal()
        bias_init = jax.nn.initializers.zeros
        self.hypernet_weight_1 = nnx.Linear(
            s_dim, n_agents * hidden_dim, kernel_init=kernel_init, rngs=rngs
        )
        self.hypernet_bias_1 = nnx.Linear(
            s_dim, hidden_dim, kernel_init=kernel_init, rngs=rngs
        )
        self.hypernet_weight_2 = nnx.Linear(
            s_dim, hidden_dim, kernel_init=kernel_init, rngs=rngs
        )
        self.hypernet_bias_2 = nnx.Sequential(
            nnx.Linear(
                s_dim,
                hidden_dim,
                kernel_init=kernel_init,
                bias_init=bias_init,
                rngs=rngs,
            ),
            nnx.relu,
            nnx.Linear(
                hidden_dim, 1, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
            ),
        )

    def __call__(self, Q: jnp.ndarray, s: jnp.ndarray):
        Q = Q.reshape(-1, 1, self.n_agents)
        W1 = jnp.abs(self.hypernet_weight_1(s))
        W1 = W1.reshape(-1, self.n_agents, self.hidden_dim)
        b1 = self.hypernet_bias_1(s)
        b1 = b1.reshape(-1, 1, self.hidden_dim)
        Q = nnx.elu(jnp.matmul(Q, W1) + b1)

        W2 = jnp.abs(self.hypernet_weight_2(s))
        W2 = W2.reshape(-1, self.hidden_dim, 1)
        b2 = self.hypernet_bias_2(s)
        b2 = b2.reshape(-1, 1, 1)
        Q_tot = jnp.matmul(Q, W2) + b2
        return Q_tot


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_space: int,
        state_space: int,
        action_space: int,
        rb_key: jax.Array,
        normalize_reward: bool = False,
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.rb_key = rb_key
        self.normalize_reward = normalize_reward
        self.episodes = [None] * buffer_size
        self.pos = 0
        self.size = 0

    def store(self, episode: dict):
        self.episodes[self.pos] = episode
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Tuple[jnp.ndarray]:
        self.rb_key, subkey = jax.random.split(self.rb_key)
        indices = jax.random.randint(subkey, (batch_size,), minval=0, maxval=self.size)
        batch = [self.episodes[i] for i in indices]
        lengths = [len(episode["obs"]) for episode in batch]
        max_length = max(lengths)
        obs = np.zeros(
            (batch_size, max_length, self.num_agents, self.obs_space), dtype=np.float32
        )
        avail_actions = np.zeros(
            (batch_size, max_length, self.num_agents, self.action_space), dtype=np.bool_
        )
        actions = np.zeros((batch_size, max_length, self.num_agents), dtype=np.int32)
        reward = np.zeros((batch_size, max_length), dtype=np.float32)
        next_obs = np.zeros(
            (batch_size, max_length, self.num_agents, self.obs_space), dtype=np.float32
        )
        states = np.zeros((batch_size, max_length, self.state_space), dtype=np.float32)
        next_states = np.zeros(
            (batch_size, max_length, self.state_space), dtype=np.float32
        )
        done = np.zeros((batch_size, max_length), dtype=np.int32)
        mask = np.zeros((batch_size, max_length), dtype=np.bool_)

        for i in range(batch_size):
            length = lengths[i]
            obs[i, :length] = np.stack(batch[i]["obs"])
            avail_actions[i, :length] = np.stack(batch[i]["avail_actions"])
            actions[i, :length] = np.stack(batch[i]["actions"])
            reward[i, :length] = np.stack(batch[i]["reward"])
            next_obs[i, :length] = np.stack(batch[i]["next_obs"])
            states[i, :length] = np.stack(batch[i]["states"])
            next_states[i, :length] = np.stack(batch[i]["next_states"])
            done[i, :length] = np.stack(batch[i]["done"])
            mask[i, :length] = 1
        (
            obs,
            actions,
            reward,
            next_obs,
            states,
            next_states,
            avail_actions,
            done,
            mask,
        ) = jax.tree.map(
            jnp.asarray,
            (
                obs,
                actions,
                reward,
                next_obs,
                states,
                next_states,
                avail_actions,
                done,
                mask,
            ),
        )
        if self.normalize_reward:
            mu = jnp.mean(reward)
            std = jnp.std(reward)
            reward = (reward - mu) / (std + 1e-6)

        return (
            obs,
            actions,
            reward,
            next_obs,
            states,
            next_states,
            avail_actions,
            done,
            mask,
        )


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def environment(
    env_type: str, env_name: str, env_family: str, agent_ids: bool, kwargs: dict
):
    if env_type == "pz":
        env = PettingZooWrapper(
            family=env_family, env_name=env_name, agent_ids=agent_ids, **kwargs
        )
    elif env_type == "smaclite":
        env = SMACliteWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    elif env_type == "lbf":
        env = LBFWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    return env


@nnx.jit
def soft_update(target_state: Any, utility_state: Any, polyak: float):
    return jax.tree.map(
        lambda t, s: polyak * s + (1.0 - polyak) * t, target_state, utility_state
    )


@nnx.jit
def select_action(
    utility_network: nnx.Module, obs: jnp.ndarray, avail_action: jnp.ndarray
):
    q_values = utility_network(x=obs, avail_action=avail_action)
    return jnp.argmax(q_values, axis=-1)


def loss_fn(
    net: nnx.Dict, target_net: nnx.Dict, batch: Tuple[jnp.ndarray], gamma: float
):
    (
        batch_obs,
        batch_action,
        batch_reward,
        batch_next_obs,
        batch_states,
        batch_next_states,
        batch_avail_action,
        batch_done,
        batch_mask,
    ) = batch
    q_next_max = target_net["target_network"](
        batch_next_obs, avail_action=batch_avail_action
    ).max(axis=-1)
    q_tot_target = target_net["target_mixer"](
        Q=q_next_max, s=batch_next_states
    ).squeeze()

    targets = batch_reward + gamma * (1 - batch_done) * q_tot_target.reshape(
        batch_done.shape
    )
    q_values = jnp.take_along_axis(
        arr=net["utility_network"](batch_obs),
        indices=jnp.expand_dims(batch_action, axis=-1),
        axis=-1,
    ).squeeze()
    q_tot = net["mixer"](Q=q_values, s=batch_states).squeeze().reshape(targets.shape)

    loss = optax.l2_loss(jax.lax.stop_gradient(targets), q_tot)
    loss = jnp.where(batch_mask, loss, 0).sum()
    loss /= batch_mask.sum()
    # batch = jax.tree.map(lambda x: jnp.moveaxis(x, 1, 0), batch)

    # def loss_step(loss_t, seq_t):
    #     (
    #         batch_obs_t,
    #         batch_action_t,
    #         batch_reward_t,
    #         batch_next_obs_t,
    #         batch_states_t,
    #         batch_next_states_t,
    #         batch_avail_action_t,
    #         batch_done_t,
    #         batch_mask_t,
    #     ) = seq_t
    #     q_next_max = target_net["target_network"](
    #         batch_next_obs_t, avail_action=batch_avail_action_t
    #     ).max(axis=-1)
    #     q_tot_target = target_net["target_mixer"](
    #         Q=q_next_max, s=batch_next_states_t
    #     ).squeeze()
    #     targets = batch_reward_t + gamma * (1 - batch_done_t) * q_tot_target
    #     q_values = jnp.take_along_axis(
    #         arr=net["utility_network"](batch_obs_t),
    #         indices=jnp.expand_dims(batch_action_t, axis=-1),
    #         axis=-1,
    #     ).squeeze()
    #     q_tot = net["mixer"](Q=q_values, s=batch_states_t).squeeze()

    #     temp_loss = optax.l2_loss(jax.lax.stop_gradient(targets), q_tot)
    #     temp_loss = jnp.where(batch_mask_t, temp_loss, 0)
    #     loss = loss_t + temp_loss.sum()
    #     return loss, None

    # loss, _ = jax.lax.scan(
    #     f=loss_step,
    #     init=0,
    #     xs=batch,
    # )
    # loss /= batch[-1].sum()
    return loss


@nnx.jit
def training_step(
    net: nnx.Dict,
    target_net: nnx.Dict,
    optimizer: nnx.Optimizer,
    batch: Tuple[jnp.ndarray],
    gamma: float,
):
    loss, grads = nnx.value_and_grad(loss_fn)(net, target_net, batch, gamma)
    g_norm = optax.global_norm(grads)
    optimizer.update(net, grads)
    return net, optimizer, loss, g_norm


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Set the randomness
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.key(seed)
    key, rb_key = jax.random.split(key)
    rngs = nnx.Rngs(seed)
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
    ## initialize the networks
    utility_network = Qnetwork(
        input_dim=env.get_obs_size(),
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layers,
        output_dim=env.get_action_size(),
        rngs=rngs,
    )
    target_network = nnx.clone(utility_network)
    mixer = MixingNetwork(
        n_agents=env.n_agents,
        s_dim=env.get_state_size(),
        hidden_dim=args.hyper_dim,
        rngs=rngs,
    )
    target_mixer = nnx.clone(mixer)

    nets = nnx.Dict({"utility_network": utility_network, "mixer": mixer})
    target_nets = nnx.Dict(
        {"target_network": target_network, "target_mixer": target_mixer}
    )
    ## initialize the optimizer
    optimizer = getattr(optax, args.optimizer)(learning_rate=args.learning_rate)
    if args.clip_gradients > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.clip_gradients), optimizer
        )
    optimizer = nnx.Optimizer(nets, optimizer, wrt=nnx.Param)

    ## initialize a shared replay buffer
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space=env.get_obs_size(),
        state_space=env.get_state_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        rb_key=rb_key,
        normalize_reward=args.normalize_reward,
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
            name=f"QMIX-JAX-{run_name}",
        )
    writer = SummaryWriter(f"runs/QMIX-JAX-{run_name}")
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
            "next_obs": [],
            "states": [],
            "next_states": [],
            "done": [],
            "avail_actions": [],
        }
        obs, _ = env.reset()
        avail_action = env.get_avail_actions()
        state = env.get_state()
        ep_reward, ep_length = 0, 0
        done, truncated = False, False
        while not done and not truncated:
            epsilon = linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * args.total_timesteps,
                step,
            )
            if random.random() < epsilon:
                actions = env.sample()
            else:
                actions = select_action(
                    utility_network=nets["utility_network"],
                    obs=jnp.asarray(obs),
                    avail_action=jnp.asarray(avail_action).astype(jnp.bool_),
                )
            next_obs, reward, done, truncated, infos = env.step(actions)
            avail_action = env.get_avail_actions()  # Get the mask of 'next_obs' and store it in the replay, we need it for the bellman loss
            next_state = env.get_state()

            ep_reward += reward
            ep_length += 1
            step += 1
            episode["obs"].append(obs)
            episode["actions"].append(actions)
            episode["reward"].append(reward)
            episode["next_obs"].append(next_obs)
            episode["done"].append(done)
            episode["avail_actions"].append(avail_action)
            episode["states"].append(state)
            episode["next_states"].append(next_state)

            obs = next_obs
            state = next_state

        rb.store(episode)
        num_episode += 1
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)
        if args.env_type == "smaclite":
            ep_stats.append(infos)  ## Add battle won for smaclite

        if num_episode > args.batch_size:
            if num_episode % args.train_freq == 0:
                batch = rb.sample(args.batch_size)
                nets, optimizer, loss, g_norm = training_step(
                    nets, target_nets, optimizer, batch, args.gamma
                )
                num_updates += 1
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/grads", g_norm.item(), step)
                writer.add_scalar("train/num_updates", num_updates, step)

            if num_episode % args.target_network_update_freq == 0:
                new_target_state = soft_update(
                    nnx.state(target_nets["target_network"]),
                    nnx.state(nets["utility_network"]),
                    args.polyak,
                )
                nnx.update(target_nets["target_network"], new_target_state)

                new_target_mixer_state = soft_update(
                    nnx.state(target_nets["target_mixer"]),
                    nnx.state(nets["mixer"]),
                    args.polyak,
                )
                nnx.update(target_nets["target_mixer"], new_target_mixer_state)
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
            while eval_ep < args.num_eval_ep:
                actions = select_action(
                    nets["utility_network"],
                    jnp.asarray(eval_obs),
                    jnp.asarray(eval_env.get_avail_actions().astype(jnp.bool)),
                )
                next_obs_, reward, done, truncated, infos = eval_env.step(actions)
                current_reward += reward
                current_ep_length += 1
                eval_obs = next_obs_
                if done or truncated:
                    eval_obs, _ = eval_env.reset()
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
    env.close()
    eval_env.close()
