from functools import partial
from typing import Tuple, Any
import jax
import optax
import jax.numpy as jnp
import flax.nnx as nnx
from flax import struct
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
    env_type: str = "smaclite"
    """ Pettingzoo, SMAClite ... """
    env_name: str = "3m"
    """ Name of the environment"""
    env_family: str = "mpe"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    buffer_size: int = 10000
    """ The size of the replay buffer"""
    seq_length: int = 10
    """ Length of the sequence to store in the buffer"""
    burn_in: int = 8
    """Sequences to burn during batch updates"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    learning_starts: int = 5000
    """ Number of env steps to initialize the replay buffer"""
    train_freq: int = 3
    """ Train the network each «train_freq» step in the environment. The used value is train_freq*num_envs"""
    optimizer: str = "adam"
    """ The optimizer"""
    learning_rate: float = 0.0005
    """ Learning rate"""
    batch_size: int = 32
    """Batch size"""
    start_e: float = 1
    """ The starting value of epsilon, for exploration"""
    end_e: float = 0.05
    """ The end value of epsilon, for exploration"""
    exploration_fraction: float = 0.05
    """ The fraction of «total-timesteps» it takes from to go from start_e to  end_e"""
    hidden_dim: int = 32
    """ Hidden dimension"""
    num_layers: int = 1
    """ Number of layers"""
    normalize_reward: bool = False
    """ Normalize the rewards"""
    target_network_update_freq: int = 1
    """ Frequency of updating target network. The used value is target_network_update_freq*num_envs"""
    polyak: float = 0.005
    """ Update the target network each target_network_update_freq» step in the environment"""
    log_every: int = 10
    """ Logging steps"""
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    eval_steps: int = 10000
    """ Evaluate the policy each eval_steps steps. The used value is eval_steps*num_envs"""
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""
    seed: int = 1
    """ Random seed"""


@struct.dataclass
class TrainConfig:
    seq_length: int
    batch_size: int
    n_agents: int
    hidden_dim: int
    burn_in: int
    gamma: float


class Qnetwork(nnx.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.gru = nnx.GRUCell(hidden_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        h: jnp.ndarray | None = None,
        avail_action: jnp.ndarray | None = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nnx.relu(self.fc1(x))
        if h is None:
            h = jnp.zeros((x.shape[0], self.hidden_dim))
        h, _ = self.gru(carry=h, inputs=x)
        x = self.fc2(nnx.relu(h))
        if avail_action is not None:
            x = jnp.where(avail_action, x, jnp.finfo(jnp.float32).min)
        return x, h


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_space: int,
        action_space: int,
        seq_length: int,
        rb_key: jax.Array,
        normalize_reward: bool = False,
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.seq_length = seq_length
        self.rb_key = rb_key
        self.normalize_reward = normalize_reward
        self.obs = np.zeros(
            (self.buffer_size, self.seq_length, self.num_agents, self.obs_space),
            dtype=np.float32,
        )
        self.action = np.zeros(
            (self.buffer_size, self.seq_length, self.num_agents), dtype=np.int32
        )
        self.reward = np.zeros((self.buffer_size, self.seq_length), dtype=np.float32)
        self.next_obs = np.zeros(
            (self.buffer_size, self.seq_length, self.num_agents, self.obs_space),
            dtype=np.float32,
        )
        self.next_avail_action = np.zeros(
            (self.buffer_size, self.seq_length, self.num_agents, self.action_space),
            dtype=np.bool_,
        )
        self.done = np.zeros((self.buffer_size, self.seq_length), dtype=np.int32)
        self.pos = 0
        self.size = 0
        self.last_pos = 0

    def store(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        done: jnp.ndarray,
        next_obs: jnp.ndarray,
        next_avail_action: jnp.ndarray,
        is_last: bool = False,
    ) -> None:
        if is_last:
            toadd = self.seq_length - len(obs)
            obs = np.concatenate((self.obs[self.last_pos][-toadd:], obs), axis=0)
            action = np.concatenate(
                (self.action[self.last_pos][-toadd:], action), axis=0
            )
            reward = np.concatenate(
                (self.reward[self.last_pos][-toadd:], reward), axis=0
            )
            done = np.concatenate((self.done[self.last_pos][-toadd:], done), axis=0)
            next_obs = np.concatenate(
                (self.next_obs[self.last_pos][-toadd:], next_obs), axis=0
            )
            next_avail_action = np.concatenate(
                (self.next_avail_action[self.last_pos][-toadd:], next_avail_action),
                axis=0,
            )
        self.obs[self.pos] = obs
        self.action[self.pos] = action
        self.reward[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.next_avail_action[self.pos] = next_avail_action
        self.done[self.pos] = done
        self.last_pos = self.pos
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Tuple[jnp.ndarray]:
        self.rb_key, subkey = jax.random.split(self.rb_key)
        indices = jax.random.randint(subkey, (batch_size,), minval=0, maxval=self.size)
        obs_batch = self.obs[indices]
        action_batch = self.action[indices]
        reward_batch = self.reward[indices]
        next_obs_batch = self.next_obs[indices]
        next_avail_action_batch = self.next_avail_action[indices]
        done_batch = self.done[indices]
        batch = (
            obs_batch,
            action_batch,
            reward_batch,
            next_obs_batch,
            next_avail_action_batch,
            done_batch,
        )
        obs, action, reward, next_obs, next_avail, done = jax.tree.map(
            jnp.asarray, batch
        )
        if self.normalize_reward:
            mu = jnp.mean(reward)
            std = jnp.std(reward)
            reward = (reward - mu) / (std + 1e-6)
        return (obs, action, reward, next_obs, next_avail, done)


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
def soft_update(target_state: Any, utility_state: Any, polyak: float) -> Any:
    return jax.tree.map(
        lambda t, s: polyak * s + (1.0 - polyak) * t, target_state, utility_state
    )


def loss_fn(
    utility_network: nnx.Module,
    target_network: nnx.Module,
    batch: Tuple[jnp.ndarray],
    train_config: TrainConfig,
):
    batch = jax.tree.map(lambda x: jnp.moveaxis(x, 1, 0), batch)
    (
        batch_obs,
        batch_action,
        batch_reward,
        batch_next_obs,
        batch_next_avail_action,
        batch_done,
    ) = batch
    batch_obs, batch_next_obs, batch_next_avail_action = jax.tree.map(
        lambda x: x.reshape(
            train_config.seq_length, train_config.batch_size * train_config.n_agents, -1
        ),
        (batch_obs, batch_next_obs, batch_next_avail_action),
    )

    def burn_in(h: jnp.ndarray, seq_t: Tuple[jnp.ndarray]):
        h_target, h_utility = h
        batch_obs_t, batch_next_obs_t = seq_t
        _, h_target = target_network(batch_next_obs_t, h=h_target)
        _, h_utility = utility_network(batch_obs_t, h=h_utility)
        return (h_target, h_utility), None

    h = jnp.zeros(
        (train_config.batch_size * train_config.n_agents, train_config.hidden_dim)
    )
    h_burnt, _ = jax.lax.scan(
        burn_in,
        (h, h),
        (batch_obs[: train_config.burn_in], batch_next_obs[: train_config.burn_in]),
    )

    def loss_t(h: jnp.ndarray, seq_t: Tuple[jnp.ndarray]):
        h_target, h_utility, temp_loss = h
        (
            batch_obs_t,
            batch_action_t,
            batch_next_obs_t,
            batch_next_avail_action_t,
            batch_reward_t,
            batch_done_t,
        ) = seq_t
        q_next, h_target = target_network(
            batch_next_obs_t, h=h_target, avail_action=batch_next_avail_action_t
        )
        q_next = q_next.reshape(train_config.batch_size, env.n_agents, -1)
        q_next_max = q_next.max(axis=-1)
        vdn_q_max = q_next_max.sum(axis=-1)
        targets = batch_reward_t + train_config.gamma * (1 - batch_done_t) * vdn_q_max
        q_values, h_utility = utility_network(batch_obs_t, h=h_utility)
        q_values = q_values.reshape(train_config.batch_size, env.n_agents, -1)
        q_values = jnp.take_along_axis(
            arr=q_values, indices=jnp.expand_dims(batch_action_t, axis=-1), axis=-1
        ).squeeze()
        vdn_q_values = q_values.sum(axis=-1)
        temp_loss = (
            temp_loss
            + optax.l2_loss(jax.lax.stop_gradient(targets), vdn_q_values).mean()
        )
        return (h_target, h_utility, temp_loss), None

    (_, _, loss), _ = jax.lax.scan(
        f=loss_t,
        init=(h_burnt[0], h_burnt[1], 0),
        xs=(
            batch_obs[train_config.burn_in :],
            batch_action[train_config.burn_in :],
            batch_next_obs[train_config.burn_in :],
            batch_next_avail_action[train_config.burn_in :],
            batch_reward[train_config.burn_in :],
            batch_done[train_config.burn_in :],
        ),
    )
    loss = loss / (train_config.seq_length - train_config.burn_in)
    return loss


@partial(jax.jit, static_argnums=(4,))
def training_step(
    utility_network: nnx.Module,
    target_network: nnx.Module,
    optimizer: nnx.Optimizer,
    batch: Tuple[jnp.ndarray],
    train_config: TrainConfig,
):
    loss, grads = nnx.value_and_grad(loss_fn)(
        utility_network, target_network, batch, train_config
    )
    g_norm = optax.global_norm(grads)
    optimizer.update(utility_network, grads)
    return utility_network, optimizer, loss, g_norm


@nnx.jit
def select_action(
    network: Qnetwork,
    obs: jnp.ndarray,
    h: jnp.ndarray | None,
    avail_action: jnp.ndarray | None,
):
    q_values, h = network(x=obs, h=h, avail_action=avail_action)
    return jnp.argmax(q_values, axis=-1), h


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Set the randomness
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.key(seed)
    key, rb_key = jax.random.split(key)
    rngs = nnx.Rngs(seed)
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
    utility_network = Qnetwork(
        input_dim=env.get_obs_size(),
        hidden_dim=args.hidden_dim,
        output_dim=env.get_action_size(),
        rngs=rngs,
    )
    target_network = nnx.clone(utility_network)

    optimizer = getattr(optax, args.optimizer)(learning_rate=args.learning_rate)
    if args.clip_gradients > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.clip_gradients), optimizer
        )
    optimizer = nnx.Optimizer(utility_network, optimizer, wrt=nnx.Param)

    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space=env.get_obs_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        seq_length=args.seq_length,
        rb_key=rb_key,
        normalize_reward=args.normalize_reward,
    )
    train_config = TrainConfig(
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        n_agents=env.n_agents,
        hidden_dim=args.hidden_dim,
        burn_in=args.burn_in,
        gamma=args.gamma,
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
            name=f"VDN-JAX-LSTM-{run_name}",
        )
    writer = SummaryWriter(f"runs/VDN-JAX-lstm-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    obs, _ = env.reset()
    avail_action = env.get_avail_actions()
    h = None
    seq_obs, seq_actions, seq_reward, seq_done, seq_next_obs, seq_next_avail_action = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    current_seq_len = 0
    ep_reward = 0
    ep_length = 0
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    num_updates = 0
    num_episodes = 0
    for step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            step,
        )
        # We always need the forward pass even when taking random actions in order to let the h flow through time
        actions, h = select_action(
            utility_network,
            jnp.asarray(obs),
            h,
            jnp.asarray(avail_action).astype(jnp.bool_),
        )
        if random.random() < epsilon:
            actions = env.sample()
        next_obs, reward, done, truncated, infos = env.step(actions)
        next_avail_action = env.get_avail_actions()  # We need the next_avail_action to compute the target loss : max of Q(next_state)

        ep_reward += reward
        ep_length += 1
        seq_obs.append(obs)
        seq_actions.append(actions)
        seq_reward.append(reward)
        seq_done.append(done)
        seq_next_obs.append(next_obs)
        seq_next_avail_action.append(next_avail_action)
        current_seq_len += 1
        obs = next_obs
        avail_action = next_avail_action
        if current_seq_len == args.seq_length:
            rb.store(
                np.stack(seq_obs),
                np.stack(seq_actions),
                np.stack(seq_reward),
                np.stack(seq_done),
                np.stack(seq_next_obs),
                np.stack(seq_next_avail_action),
            )
            current_seq_len = 0
            (
                seq_obs,
                seq_actions,
                seq_reward,
                seq_done,
                seq_next_obs,
                seq_next_avail_action,
            ) = [], [], [], [], [], []

        if done or truncated:
            obs, _ = env.reset()
            avail_action = env.get_avail_actions()
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            if args.env_type == "smaclite":
                ep_stats.append(infos)
            ep_reward = 0
            ep_length = 0
            num_episodes += 1
            h = None
            if 0 < current_seq_len and current_seq_len < args.seq_length:
                current_seq_len = 0
                rb.store(
                    np.stack(seq_obs),
                    np.stack(seq_actions),
                    np.stack(seq_reward),
                    np.stack(seq_done),
                    np.stack(seq_next_obs),
                    np.stack(seq_next_avail_action),
                    is_last=True,
                )
                (
                    seq_obs,
                    seq_actions,
                    seq_reward,
                    seq_done,
                    seq_next_obs,
                    seq_next_avail_action,
                ) = [], [], [], [], [], []

        if step > args.learning_starts:
            if step % args.train_freq == 0:
                batch = rb.sample(args.batch_size)
                utility_network, optimizer, loss, g_norm = training_step(
                    utility_network, target_network, optimizer, batch, train_config
                )
                num_updates += 1
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/grads", g_norm.item(), step)
                writer.add_scalar("train/num_updates", num_updates, step)

            if step % args.target_network_update_freq == 0:
                new_target_state = soft_update(
                    nnx.state(target_network), nnx.state(utility_network), args.polyak
                )
                nnx.update(target_network, new_target_state)

        if len(ep_rewards) > args.log_every:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            writer.add_scalar("rollout/epsilon", epsilon, step)
            writer.add_scalar("rollout/num_episodes", num_episodes, step)
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "rollout/battle_won",
                    np.mean([info["battle_won"] for info in ep_stats]),
                    step,
                )
            ep_rewards = []
            ep_lengths = []
            ep_stats = []
        if step % args.eval_steps == 0 and step > args.learning_starts:
            eval_obs, _ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            h_eval = None
            while eval_ep < args.num_eval_ep:
                actions, h_eval = select_action(
                    utility_network,
                    jnp.asarray(eval_obs),
                    h_eval,
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

    writer.close()
    if args.use_wnb:
        wandb.finish()
    env.close()
    eval_env.close()
