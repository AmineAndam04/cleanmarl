from multiprocessing import Pipe, Process
from functools import partial
from typing import Tuple
import jax
import optax
import jax.numpy as jnp
import flax.nnx as nnx
from flax import struct
import tyro
import datetime
import random
import numpy as np
from dataclasses import dataclass
from env.pettingzoo_wrapper import PettingZooWrapper
from env.smaclite_wrapper import SMACliteWrapper
from env.lbf import LBFWrapper
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="


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
    batch_size: int = 3
    """ Number of episodes to collect in each rollout"""
    actor_hidden_dim: int = 32
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 32
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    optimizer: str = "adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0008
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0008
    """ Learning rate for the critic"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    td_lambda: float = 0.95
    """ TD(λ) discount factor"""
    normalize_reward: bool = True
    """ Normalize the rewards if True"""
    normalize_advantage: bool = False
    """ Normalize the advantage if True"""
    normalize_return: bool = False
    """ Normalize the returns if True"""
    ppo_clip: float = 0.2
    """ PPO clipping factor """
    entropy_coef: float = 0.001
    """ Entropy coefficient """
    epochs: int = 3
    """ Number of training epochs"""
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    log_every: int = 10
    """ Logging steps """
    eval_steps: int = 50
    """ Evaluate the policy each «eval_steps» training steps"""
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
    gamma: float
    td_lambda: float
    ppo_clip: float
    entropy_coef: float


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_space: int,
        state_space: int,
        action_space: int,
        normalize_reward: bool = False,
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.normalize_reward = normalize_reward
        self.episodes = [None] * buffer_size
        self.pos = 0

    def add(self, episode):
        self.episodes[self.pos] = episode
        self.pos += 1

    def get_batch(self):
        self.pos = 0
        lengths = [len(episode["obs"]) for episode in self.episodes]
        max_length = max(lengths)
        obs = np.zeros(
            (self.buffer_size, max_length, self.num_agents, self.obs_space),
            dtype=np.float32,
        )
        avail_actions = np.zeros(
            (self.buffer_size, max_length, self.num_agents, self.action_space),
            dtype=np.bool_,
        )
        actions = np.zeros(
            (self.buffer_size, max_length, self.num_agents), dtype=np.int32
        )
        log_probs = np.zeros(
            (self.buffer_size, max_length, self.num_agents), dtype=np.float32
        )
        reward = np.zeros((self.buffer_size, max_length), dtype=np.float32)
        states = np.zeros(
            (self.buffer_size, max_length, self.state_space), dtype=np.float32
        )
        done = np.zeros((self.buffer_size, max_length), dtype=np.int32)
        mask = np.zeros((self.buffer_size, max_length), dtype=np.bool_)
        for i in range(self.buffer_size):
            length = lengths[i]
            obs[i, :length] = np.stack(self.episodes[i]["obs"])
            avail_actions[i, :length] = np.stack(self.episodes[i]["avail_actions"])
            actions[i, :length] = np.stack(self.episodes[i]["actions"])
            log_probs[i, :length] = np.stack(self.episodes[i]["log_prob"])
            reward[i, :length] = np.stack(self.episodes[i]["reward"])
            states[i, :length] = np.stack(self.episodes[i]["states"])
            done[i, :length] = np.stack(self.episodes[i]["done"])
            mask[i, :length] = 1

        obs, actions, log_probs, reward, states, avail_actions, done, mask = (
            jax.tree.map(
                jnp.asarray,
                (obs, actions, log_probs, reward, states, avail_actions, done, mask),
            )
        )
        if self.normalize_reward:
            mu = jnp.mean(reward[mask])
            std = jnp.std(reward[mask])
            reward = (reward - mu) / (std + 1e-6)

        self.episodes = [None] * self.buffer_size
        return (obs, actions, log_probs, reward, states, avail_actions, done, mask)


class Actor(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layer: int,
        output_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
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

    def __call__(
        self,
        x: jnp.ndarray,
        act_key: jax.Array,
        avail_action: jnp.ndarray | None = None,
    ):
        logits = self.logits(x, avail_action)
        action = jax.random.categorical(key=act_key, logits=logits)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_prop = jnp.take_along_axis(
            arr=log_probs, indices=jnp.expand_dims(action, axis=-1), axis=-1
        ).squeeze()
        return action, log_prop

    def logits(self, x: jnp.ndarray, avail_action: jnp.ndarray | None = None):
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = jnp.where(avail_action, x, -1e10)
        return x


class Critic(nnx.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_layer: int, *, rngs: nnx.Rngs
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
            nnx.Linear(hidden_dim, 1, kernel_init=kernel_init, rngs=rngs)
        )

    def __call__(self, x: jnp.ndarray):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()


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


@nnx.jit
def select_action(
    actor: nnx.Module,
    obs: jnp.ndarray,
    act_key: jax.Array,
    avail_action: jnp.ndarray | None = None,
):
    actions, log_probs = actor(x=obs, act_key=act_key, avail_action=avail_action)
    return actions, log_probs


@partial(jax.jit, static_argnums=(2, 3))
def td_lambda_advantage(
    critic: nnx.Module,
    batch: Tuple[jnp.ndarray],
    ep_lens: tuple,
    train_config: TrainConfig,
):
    b_obs, b_reward = batch
    return_lambda = jnp.zeros(b_obs.shape[:-1], dtype=jnp.float32)
    advantages = jnp.zeros(b_obs.shape[:-1], dtype=jnp.float32)
    for ep_idx in range(return_lambda.shape[0]):
        ep_len = ep_lens[ep_idx]
        next_values = critic(b_obs[ep_idx, :ep_len])

        def tb_lambda_step(carry, x):
            last_return_lambda = carry
            next_value, reward_t = x
            return_lambda_t = reward_t + train_config.gamma * (
                train_config.td_lambda * last_return_lambda
                + (1 - train_config.td_lambda) * next_value
            )
            return return_lambda_t, return_lambda_t

        xs = (
            jnp.concatenate((next_values[1:], jnp.zeros((1, next_values.shape[-1])))),
            b_reward[ep_idx, :ep_len],
        )
        _, return_lambda_ep = jax.lax.scan(
            f=tb_lambda_step,
            init=jnp.zeros(b_obs.shape[-2]),
            xs=xs,
            reverse=True,
        )
        return_lambda = return_lambda.at[ep_idx, :ep_len].set(return_lambda_ep)
        advantages = advantages.at[ep_idx, :ep_len].set(
            return_lambda[ep_idx, :ep_len] - next_values
        )
    ##
    # return_lambda = jnp.zeros(b_obs.shape[:-1], dtype=jnp.float32)
    # advantages = jnp.zeros(b_obs.shape[:-1], dtype=jnp.float32)
    # for ep_idx in range(return_lambda.shape[0]):
    #     ep_len = ep_lens[ep_idx]
    #     next_values = critic(
    #         jax.lax.dynamic_slice_in_dim(b_obs[int(ep_idx)], 1, (ep_len - 1), axis=0)
    #     )

    #     def tb_lambda_step(carry, x):
    #         last_return_lambda = carry
    #         next_value, reward_t = x
    #         return_lambda_t = reward_t + train_config.gamma * (
    #             train_config.td_lambda * last_return_lambda
    #             + (1 - train_config.td_lambda) * next_value
    #         )
    #         return return_lambda_t, return_lambda_t

    #     xs = (
    #         next_values,
    #         jax.lax.dynamic_slice_in_dim(b_reward[int(ep_idx)], 0, ep_len - 1, axis=0),
    #     )
    #     _, return_lambda_ep = jax.lax.scan(
    #         f=tb_lambda_step,
    #         init=jnp.ones(b_obs.shape[-2]) * b_reward[ep_idx][ep_len - 1],
    #         xs=xs,
    #         reverse=True,
    #     )
    #     return_lambda = return_lambda.at[ep_idx, : ep_len - 1].set(return_lambda_ep)
    #     return_lambda = return_lambda.at[ep_idx, ep_len - 1].set(
    #         b_reward[ep_idx][ep_len - 1]
    #     )
    #     advantages = advantages.at[ep_idx, :ep_len].set(
    #         return_lambda[ep_idx, :ep_len] - critic(b_obs[ep_idx, :ep_len])
    #     )
    return jax.lax.stop_gradient(return_lambda), jax.lax.stop_gradient(advantages)


def actor_loss(
    actor: nnx.Module,
    batch: Tuple[jnp.ndarray],
    advantages: jnp.ndarray,
    train_config: TrainConfig,
):
    b_obs, b_actions, b_log_probs, b_avail_actions, b_mask = batch
    current_logits = actor.logits(x=b_obs, avail_action=b_avail_actions)
    log_probs = jax.nn.log_softmax(current_logits, axis=-1)
    current_logprob = jnp.take_along_axis(
        arr=log_probs, indices=jnp.expand_dims(b_actions, axis=-1), axis=-1
    ).squeeze()
    log_ratio = current_logprob - b_log_probs
    ratio = jnp.exp(log_ratio)
    pg_loss1 = advantages * ratio
    pg_loss2 = advantages * jax.lax.clamp(
        1 - train_config.ppo_clip, ratio, 1 + train_config.ppo_clip
    )
    pg_loss = jax.lax.min(pg_loss1, pg_loss2).mean(axis=-1)
    pg_loss = jnp.where(b_mask, pg_loss, 0).sum()
    entropy_loss = -(jnp.exp(log_probs) * log_probs).mean(axis=-1)
    entropy_loss = jnp.where(b_mask[:, :, None], entropy_loss, 0).sum()
    entropies = entropy_loss

    ac_loss = -pg_loss - train_config.entropy_coef * entropy_loss
    kl_divergence = jnp.where(b_mask, ((ratio - 1) - log_ratio).mean(axis=-1), 0).sum()
    clipped_ratio = jnp.where(
        b_mask, (jnp.absolute(ratio - 1.0) > train_config.ppo_clip).mean(axis=-1), 0
    ).sum()

    # def ppo_loss_t(carry, batch_t):
    #     ac_loss, entropies, kl_divergence, clipped_ratio = carry
    #     obs_t, actions_t, log_probs_t, avail_actions_t, mask_t, advantages_t = batch_t
    #     current_logits = actor.logits(x=obs_t, avail_action=avail_actions_t)
    #     log_probs = jax.nn.log_softmax(current_logits, axis=-1)
    #     current_logprob = jnp.take_along_axis(
    #         arr=log_probs, indices=jnp.expand_dims(actions_t, axis=-1), axis=-1
    #     ).squeeze()
    #     log_ratio = current_logprob - log_probs_t
    #     ratio = jnp.exp(log_ratio)

    #     pg_loss1 = advantages_t * ratio
    #     pg_loss2 = advantages_t * jax.lax.clamp(
    #         1 - train_config.ppo_clip, ratio, 1 + train_config.ppo_clip
    #     )

    #     pg_loss = jax.lax.min(pg_loss1, pg_loss2).mean(axis=-1)
    #     pg_loss = jnp.where(mask_t, pg_loss, 0).sum()
    #     entropy_loss = -(jnp.exp(log_probs) * log_probs).mean(axis=-1)
    #     entropy_loss = jnp.where(mask_t[:, None], entropy_loss, 0).sum()
    #     entropies += entropy_loss

    #     ac_loss += -pg_loss - train_config.entropy_coef * entropy_loss
    #     kl_divergence_t = jnp.where(
    #         mask_t, ((ratio - 1) - log_ratio).mean(axis=-1), 0
    #     ).sum()
    #     kl_divergence += kl_divergence_t
    #     clipped_ratio_t = jnp.where(
    #         mask_t, (jnp.absolute(ratio - 1.0) > train_config.ppo_clip).mean(axis=-1), 0
    #     ).sum()
    #     clipped_ratio += clipped_ratio_t
    #     return (ac_loss, entropies, kl_divergence, clipped_ratio), None

    # (ac_loss, entropies, kl_divergence, clipped_ratio), _ = jax.lax.scan(
    #     f=ppo_loss_t,
    #     init=(0, 0, 0, 0),
    #     xs=jax.tree.map(lambda x: jnp.moveaxis(x, 1, 0), (*batch, advantages)),
    # )
    return ac_loss / batch[-1].sum(), (
        entropies / batch[-1].sum(),
        kl_divergence / batch[-1].sum(),
        clipped_ratio / batch[-1].sum(),
    )


@nnx.jit
def actor_training_step(
    actor: nnx.Module,
    actor_optimizer: nnx.Optimizer,
    batch: Tuple[jnp.ndarray],
    advantages: jnp.ndarray,
    train_config: TrainConfig,
):
    (ac_loss, (entropies, kl_divergence, clipped_ratio)), grads = nnx.value_and_grad(
        actor_loss, has_aux=True
    )(actor, batch, advantages, train_config)
    g_norm = optax.global_norm(grads)
    actor_optimizer.update(actor, grads)
    return (
        actor,
        actor_optimizer,
        ac_loss,
        entropies,
        kl_divergence,
        clipped_ratio,
        g_norm,
    )


def critic_loss(
    critic: nnx.Module, batch: Tuple[jnp.ndarray], return_lambda: jnp.ndarray
):
    def critic_loss_t(carry, batch_t):
        obs_t, mask_t, return_lambda_t = batch_t
        current_values = critic(obs_t)
        value_loss = optax.l2_loss(return_lambda_t, current_values)
        value_loss = jnp.where(mask_t, value_loss.mean(axis=-1), 0).sum()
        carry += value_loss
        return carry, None

    cr_loss, _ = jax.lax.scan(f=critic_loss_t, init=0, xs=(*batch, return_lambda))
    return cr_loss / batch[-1].sum()


@nnx.jit
def critic_training_step(
    critic: nnx.Module,
    critic_optimizer: nnx.Optimizer,
    batch: Tuple[jnp.ndarray],
    return_lambda: jnp.ndarray,
):
    loss, grads = nnx.value_and_grad(critic_loss)(critic, batch, return_lambda)
    g_norm = optax.global_norm(grads)
    critic_optimizer.update(critic, grads)
    return critic, critic_optimizer, loss, g_norm


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = tyro.cli(Args)
    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.key(seed)
    key, act_key = jax.random.split(key)
    rngs = nnx.Rngs(seed)
    ## import the environment
    kwargs = {}  # {"render_mode":'human',"shared_reward":False}
    conns = [Pipe() for _ in range(args.batch_size)]
    ippo_conns, env_conns = zip(*conns)
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
        for _ in range(args.batch_size)
    ]
    processes = [
        Process(target=env_worker, args=(env_conns[i], envs[i]))
        for i in range(args.batch_size)
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

    ## Initialize the actor, critic and target-critic networks
    actor = Actor(
        input_dim=eval_env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=eval_env.get_action_size(),
        rngs=rngs,
    )
    critic = Critic(
        input_dim=eval_env.get_obs_size(),
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        rngs=rngs,
    )

    # Optimizers
    actor_optimizer = getattr(optax, args.optimizer)(
        learning_rate=args.learning_rate_actor
    )
    critic_optimizer = getattr(optax, args.optimizer)(
        learning_rate=args.learning_rate_critic
    )
    if args.clip_gradients > 0:
        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(args.clip_gradients), actor_optimizer
        )
        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(args.clip_gradients), critic_optimizer
        )

    actor_optimizer = nnx.Optimizer(actor, actor_optimizer, wrt=nnx.Param)
    critic_optimizer = nnx.Optimizer(critic, critic_optimizer, wrt=nnx.Param)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"IPPO-JAX-multienvs-{run_name}",
        )
    writer = SummaryWriter(f"runs/IPPO-JAX-multienvs-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    rb = RolloutBuffer(
        buffer_size=args.batch_size,
        obs_space=eval_env.get_obs_size(),
        state_space=eval_env.get_state_size(),
        action_space=eval_env.get_action_size(),
        num_agents=eval_env.n_agents,
        normalize_reward=args.normalize_reward,
    )
    train_config = TrainConfig(
        gamma=args.gamma,
        td_lambda=args.td_lambda,
        ppo_clip=args.ppo_clip,
        entropy_coef=args.entropy_coef,
    )
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    training_step = 0
    num_episodes = 0
    step = 0
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
            for _ in range(args.batch_size)
        ]

        for ippo_conn in ippo_conns:
            ippo_conn.send(("reset", None))

        contents = [ippo_conn.recv() for ippo_conn in ippo_conns]
        obs = np.stack([content["obs"] for content in contents], axis=0)
        avail_action = np.stack(
            [content["avail_actions"] for content in contents], axis=0
        )
        state = np.stack([content["state"] for content in contents])
        alive_envs = list(range(args.batch_size))
        ep_reward, ep_length, ep_stat = (
            [0] * args.batch_size,
            [0] * args.batch_size,
            [0] * args.batch_size,
        )
        while len(alive_envs) > 0:
            key, act_key = jax.random.split(key)
            actions, log_probs = select_action(
                actor,
                jnp.asarray(obs),
                act_key,
                avail_action=jnp.asarray(avail_action).astype(jnp.bool_),
            )
            if log_probs.ndim >= 2:
                log_probs = log_probs.squeeze()
            else:
                log_probs = log_probs.reshape(actions.shape)
            for i, j in enumerate(alive_envs):
                ippo_conns[j].send(("step", actions[i]))
            contents = [ippo_conns[i].recv() for i in alive_envs]
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

            obs = []
            state = []
            avail_action = []
            for i, j in enumerate(alive_envs[:]):
                if done[i] or truncated[i]:
                    alive_envs.remove(j)
                    rb.add(episodes[j])
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

        num_episodes += args.batch_size
        ep_rewards.extend(ep_reward)
        ep_lengths.extend(ep_length)
        if args.env_type == "smaclite":
            ep_stats.extend([info["battle_won"] for info in ep_stat])

        ## logging
        if len(ep_rewards) > args.log_every:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            writer.add_scalar("rollout/num_episodes", num_episodes, step)
            if args.env_type == "smaclite":
                writer.add_scalar("rollout/battle_won", np.mean(ep_stats), step)
            ep_rewards = []
            ep_lengths = []
            ep_stats = []
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
        #####  Compute TD(λ) using "Reconciling λ-Returns with Experience Replay"(https://arxiv.org/pdf/1810.09967 Equation 3)
        #####  Compute the advantage using A(s,a) = λ-Returns -V(s), see page 47 in David Silver's lecture n 4 (https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-4-model-free-prediction-.pdf)
        ep_lens = jnp.ravel(b_mask.sum(axis=-1)).tolist()
        return_lambda, advantages = td_lambda_advantage(
            critic, (b_obs, b_reward), tuple(ep_lens), train_config
        )
        if args.normalize_advantage:
            adv_mu = advantages.mean(axis=-1)[b_mask].mean()
            adv_std = advantages.mean(axis=-1)[b_mask].std()
            advantages = (advantages - adv_mu) / adv_std
        if args.normalize_return:
            ret_mu = return_lambda.mean(axis=-1)[b_mask].mean()
            ret_std = return_lambda.mean(axis=-1)[b_mask].std()
            return_lambda = (return_lambda - ret_mu) / ret_std
        actor_losses = []
        critic_losses = []
        entropies_bonuses = []
        kl_divergences = []
        actor_gradients = []
        critic_gradients = []
        clipped_ratios = []

        for _ in range(args.epochs):
            (
                actor,
                actor_optimizer,
                ac_loss,
                entropies,
                kl_divergence,
                clipped_ratio,
                actor_gradient,
            ) = actor_training_step(
                actor,
                actor_optimizer,
                (b_obs, b_actions, b_log_probs, b_avail_actions, b_mask),
                advantages,
                train_config,
            )
            critic, critic_optimizer, cr_loss, critic_gradient = critic_training_step(
                critic, critic_optimizer, (b_obs, b_mask), return_lambda
            )
            actor_losses.append(ac_loss)
            critic_losses.append(cr_loss)
            entropies_bonuses.append(entropies)
            kl_divergences.append(kl_divergence)
            actor_gradients.append(actor_gradient)
            critic_gradients.append(critic_gradient)
            clipped_ratios.append(clipped_ratio)
            training_step += 1
        writer.add_scalar("train/critic_loss", np.mean(critic_losses), step)
        writer.add_scalar("train/actor_loss", np.mean(actor_losses), step)
        writer.add_scalar("train/entropy", np.mean(entropies_bonuses), step)
        writer.add_scalar("train/kl_divergence", np.mean(kl_divergences), step)
        writer.add_scalar("train/clipped_ratios", np.mean(clipped_ratios), step)
        writer.add_scalar("train/actor_gradients", np.mean(actor_gradients), step)
        writer.add_scalar("train/critic_gradients", np.mean(critic_gradients), step)
        writer.add_scalar("train/num_updates", training_step, step)

        if (training_step / args.epochs) % args.eval_steps == 0:
            eval_obs, _ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            while eval_ep < args.num_eval_ep:
                key, act_key = jax.random.split(key)
                actions, _ = select_action(
                    actor,
                    jnp.asarray(eval_obs),
                    act_key,
                    avail_action=jnp.asarray(eval_env.get_avail_actions()).astype(
                        jnp.bool_
                    ),
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
                    np.mean(np.mean([info["battle_won"] for info in eval_ep_stats])),
                    step,
                )

    writer.close()
    if args.use_wnb:
        wandb.finish()
    eval_env.close()
    for conn in ippo_conns:
        conn.send(("close", None))
    for process in processes:
        process.join()
