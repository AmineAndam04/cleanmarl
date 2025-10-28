from functools import partial
from typing import Tuple, Any
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
from env.common_interface import CommonInterface
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
    batch_size: int = 3
    """ Number of episodes to collect in each rollout"""
    actor_hidden_dim: int = 32
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 128
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
    td_lambda: float = 0.8
    """ TD(λ) discount factor"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    normalize_advantage: bool = True
    """ Normalize the advantage if True"""
    normalize_return: bool = False
    """ Normalize the returns if True"""
    target_network_update_freq: int = 1
    """ Update the target network each target_network_update_freq» step in the environment"""
    polyak: float = 0.01
    """ Polyak coefficient when using polyak averaging for target network update"""
    entropy_coef: float = 0.001
    """ Entropy coefficient """
    start_e: float = 0.5
    """ The starting value of epsilon. See Architecture & Training in COMA's paper Sec. 5"""
    end_e: float = 0.002
    """ The end value of epsilon. See Architecture & Training in COMA's paper Sec. 5"""
    exploration_fraction: float = 750
    """ The number of training steps it takes from to go from start_e to  end_e"""
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    log_every: int = 10
    """ Log rollout stats every log_every episode"""
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
    seed: int = 1
    """ Random seed"""


@struct.dataclass
class TrainConfig:
    gamma: float
    td_lambda: float
    normalize_advantage: bool
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

    def add(self, episode: dict):
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
            reward[i, :length] = np.stack(self.episodes[i]["reward"])
            states[i, :length] = np.stack(self.episodes[i]["states"])
            done[i, :length] = np.stack(self.episodes[i]["done"])
            mask[i, :length] = 1
        obs, avail_actions, actions, reward, states, done, mask = jax.tree.map(
            jnp.asarray, (obs, avail_actions, actions, reward, states, done, mask)
        )
        if self.normalize_reward:
            mu = jnp.mean(reward)
            std = jnp.std(reward)
            reward = (reward - mu) / (std + 1e-6)

        self.episodes = [None] * self.buffer_size
        return (obs, avail_actions, actions, reward, states, done, mask)


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

    def __call__(
        self,
        x: jnp.ndarray,
        act_key: jax.Array,
        avail_action: jnp.ndarray,
        eps: float = 0.0,
    ):
        logits = self.logits(x, eps, avail_action)

        actions = jax.random.categorical(key=act_key, logits=logits)
        return actions

    def logits(self, x: jnp.ndarray, avail_action: jnp.ndarray, eps: float = 0.0):
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = jnp.where(avail_action, x, jnp.finfo(jnp.float32).min)
        masked_eps = (avail_action) * (eps / avail_action.sum(axis=-1, keepdims=True))
        probs = (1 - eps) * nnx.softmax(x, axis=-1) + masked_eps
        return jnp.log(probs + 1e-8)


class Critic(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layer: int,
        output_dim: int,
        num_agents: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.input_dim = input_dim
        self.output_dim = output_dim

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
        state: jnp.ndarray,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        avail_actions: jnp.ndarray | None = None,
    ):
        if state.ndim < 2:
            state, observations, actions = jax.tree.map(
                lambda x: jnp.expand_dims(x, axis=0), (state, observations, actions)
            )
            if avail_actions is not None:
                avail_actions = jnp.expand_dims(avail_actions, axis=0)
        x = self.coma_inputs(state, observations, actions)
        for layer in self.layers:
            x = layer(x)
        if avail_actions is not None:
            x = jnp.where(avail_actions, x, jnp.finfo(jnp.float32).min)
        return x.squeeze()

    def coma_inputs(
        self, state: jnp.ndarray, observations: jnp.ndarray, actions: jnp.ndarray
    ):
        coma_inputs = jnp.zeros((state.shape[0], self.num_agents, self.input_dim))
        coma_inputs = coma_inputs.at[:, :, : state.shape[-1]].set(
            jnp.expand_dims(state, axis=1)
        )

        coma_inputs = coma_inputs.at[
            :, :, state.shape[-1] : state.shape[-1] + observations.shape[-1]
        ].set(observations)

        one_hot = jax.nn.one_hot(actions.astype(jnp.int32), num_classes=self.output_dim)
        oh = jnp.broadcast_to(
            array=jnp.expand_dims(one_hot, axis=1),
            shape=(state.shape[0], self.num_agents, self.num_agents, self.output_dim),
        )
        oh = oh.reshape(
            state.shape[0], self.num_agents, (self.num_agents * self.output_dim)
        )
        oh = jnp.moveaxis(oh, 1, 0)
        cleaned_oh = []
        for agent_idx in range(1, self.num_agents + 1):
            before = jnp.expand_dims(
                oh[agent_idx - 1, :, : (agent_idx - 1) * self.output_dim], axis=0
            )
            after = jnp.expand_dims(
                oh[agent_idx - 1, :, agent_idx * self.output_dim :], axis=0
            )
            cleaned_oh.append(jnp.concatenate([before, after], axis=-1))
        oh = jnp.concatenate(cleaned_oh)
        oh = jnp.moveaxis(oh, 1, 0)
        coma_inputs = coma_inputs.at[
            :, :, state.shape[-1] + observations.shape[-1] :
        ].set(oh)
        return coma_inputs


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


def get_coma_critic_input_dim(env: CommonInterface):
    critic_input_dim = (
        env.get_obs_size()
        + env.get_state_size()
        + (env.n_agents - 1) * env.get_action_size()
    )
    return critic_input_dim


@nnx.jit
def soft_update(target_state: Any, utility_state: Any, polyak: Any):
    return jax.tree.map(
        lambda t, s: polyak * s + (1.0 - polyak) * t, target_state, utility_state
    )


@partial(jax.jit, static_argnums=(2, 3))
def td_lambda(
    target_critic: nnx.Module,
    batch: Tuple[jnp.ndarray],
    ep_lens: list,
    train_config: TrainConfig,
):
    b_obs, b_avail_actions, b_actions, b_reward, b_states, _, _ = batch
    return_lambda = jnp.zeros_like(b_actions, dtype=jnp.float32)
    for ep_idx in range(return_lambda.shape[0]):
        ep_len = ep_lens[ep_idx]

        def tb_lambda_step(carry, next_step):
            last_return_lambda = carry
            next_states, next_obs, next_actions, next_avail_action, reward_t = next_step
            next_action_value = target_critic(
                state=next_states,
                observations=next_obs,
                actions=next_actions,
                avail_actions=next_avail_action,
            )
            next_action_value = jnp.take_along_axis(
                arr=next_action_value,
                indices=jnp.expand_dims(next_actions, axis=-1),
                axis=-1,
            ).squeeze()
            return_lambda_t = reward_t + train_config.gamma * (
                train_config.td_lambda * last_return_lambda
                + (1 - train_config.td_lambda) * next_action_value
            )
            return return_lambda_t, return_lambda_t

        xs = (
            jax.lax.dynamic_slice_in_dim(
                b_states[int(ep_idx)], 1, (ep_len - 1), axis=0
            ),
            jax.lax.dynamic_slice_in_dim(b_obs[int(ep_idx)], 1, (ep_len - 1), axis=0),
            jax.lax.dynamic_slice_in_dim(
                b_actions[int(ep_idx)], 1, (ep_len - 1), axis=0
            ),
            jax.lax.dynamic_slice_in_dim(
                b_avail_actions[int(ep_idx)], 1, ep_len - 1, axis=0
            ),
            jax.lax.dynamic_slice_in_dim(b_reward[int(ep_idx)], 0, ep_len - 1, axis=0),
        )
        _, return_lambda_ep = jax.lax.scan(
            f=tb_lambda_step,
            init=jnp.ones(b_actions.shape[-1]) * b_reward[ep_idx][ep_len - 1],
            xs=xs,
            reverse=True,
        )
        return_lambda = return_lambda.at[ep_idx, : ep_len - 1].set(return_lambda_ep)
        return_lambda = return_lambda.at[ep_idx, ep_len - 1].set(
            b_reward[ep_idx][ep_len - 1]
        )
    return return_lambda


@nnx.jit
def select_action(
    actor: nnx.Module,
    obs: jnp.ndarray,
    avail_action: jnp.ndarray | None,
    eps: float,
    act_key: jax.Array,
):
    actions = actor(x=obs, act_key=act_key, eps=eps, avail_action=avail_action)
    return actions


def critic_loss(
    critic: nnx.Module, batch: Tuple[jnp.ndarray], return_lambda: jnp.ndarray
):
    def cr_loss_t(carry, batch_t):
        states_t, obs_t, actions_t, mask_t, return_lambda_t = batch_t
        tq_values = critic(state=states_t, observations=obs_t, actions=actions_t)
        tq_values = jnp.take_along_axis(
            arr=tq_values, indices=jnp.expand_dims(actions_t, axis=-1), axis=-1
        ).squeeze()
        loss_t = optax.l2_loss(jax.lax.stop_gradient(return_lambda_t), tq_values)
        loss_t = jnp.where(mask_t, loss_t, 0).sum()
        loss = carry + loss_t
        return loss, None

    cr_loss, _ = jax.lax.scan(
        f=cr_loss_t,
        init=0,
        xs=jax.tree.map(lambda x: jnp.moveaxis(x, 1, 0), (*batch, return_lambda)),
    )
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


def actor_loss(
    actor: nnx.Module,
    critic: nnx.Module,
    batch: Tuple[jnp.ndarray],
    train_config: TrainConfig,
):
    def actor_loss_t(carry, batch_t):
        obs_t, avail_actions_t, states_t, actions_t, mask_t = batch_t
        log_pi = actor.logits(obs_t, avail_action=avail_actions_t)
        pi = jnp.exp(log_pi)
        entropy_loss = -(pi * log_pi).mean(axis=-1)
        entropy_loss = jnp.where(mask_t, entropy_loss, 0)
        entropy_loss = entropy_loss.sum()
        q_values = critic(state=states_t, observations=obs_t, actions=actions_t)
        coma_baseline = pi * q_values
        coma_baseline = coma_baseline.sum(axis=-1)
        current_q = jnp.take_along_axis(
            arr=q_values, indices=jnp.expand_dims(actions_t, axis=-1), axis=-1
        ).squeeze()
        advantage = jax.lax.stop_gradient(current_q - coma_baseline)
        if train_config.normalize_advantage:
            masked_advantage = jnp.where(mask_t, advantage.mean(axis=-1), 0)
            adv_mean = masked_advantage.sum() / mask_t.sum()
            adv_std = jnp.sqrt(
                jnp.where(mask_t, ((advantage - adv_mean) ** 2).mean(axis=-1), 0).sum()
                / mask_t.sum()
            )
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)
        log_pi = jnp.take_along_axis(
            arr=log_pi, indices=jnp.expand_dims(actions_t, axis=-1), axis=-1
        ).squeeze()
        loss_t = log_pi * advantage
        loss_t = jnp.where(mask_t, loss_t, 0).sum()
        loss_t = -loss_t - train_config.entropy_coef * entropy_loss
        loss, entropies = carry
        loss += loss_t
        entropies += entropy_loss
        return (loss, entropies), None

    (ac_loss, entropies), _ = jax.lax.scan(
        f=actor_loss_t,
        init=(0, 0),
        xs=jax.tree.map(lambda x: jnp.moveaxis(x, 1, 0), batch),
    )
    return ac_loss / batch[-1].sum(), entropies / batch[-1].sum()


@partial(jax.jit, static_argnums=4)
def actor_training_step(
    actor: nnx.Module,
    critic: nnx.Module,
    actor_optimizer: nnx.Optimizer,
    batch: Tuple[jnp.ndarray],
    train_config: TrainConfig,
):
    (loss, entropies), grads = nnx.value_and_grad(actor_loss, has_aux=True)(
        actor, critic, batch, train_config
    )
    g_norm = optax.global_norm(grads)
    actor_optimizer.update(actor, grads)
    return actor, actor_optimizer, loss, entropies, g_norm


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Keys
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

    ## Initialize the actor, critic and target-critic networks
    actor = Actor(
        input_dim=env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=env.get_action_size(),
        rngs=rngs,
    )
    critic_input_dim = get_coma_critic_input_dim(env)
    critic = Critic(
        input_dim=critic_input_dim,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        output_dim=env.get_action_size(),
        num_agents=env.n_agents,
        rngs=rngs,
    )
    target_critic = nnx.clone(critic)
    # optimizer
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
            name=f"COMA-JAX-{run_name}",
        )
    writer = SummaryWriter(f"runs/COMA-JAX-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    rb = RolloutBuffer(
        buffer_size=args.batch_size,
        obs_space=env.get_obs_size(),
        state_space=env.get_state_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        normalize_reward=args.normalize_reward,
    )
    train_config = TrainConfig(
        gamma=args.gamma,
        td_lambda=args.td_lambda,
        normalize_advantage=args.normalize_advantage,
        entropy_coef=args.entropy_coef,
    )
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    step = 0
    training_step = 0
    while step < args.total_timesteps:
        num_episode = 0
        epsilon = linear_schedule(
            args.start_e, args.end_e, args.exploration_fraction, training_step
        )
        while num_episode < args.batch_size:
            episode = {
                "obs": [],
                "actions": [],
                "reward": [],
                "states": [],
                "done": [],
                "avail_actions": [],
            }
            obs, _ = env.reset()
            ep_reward, ep_length = 0, 0
            done, truncated = False, False
            while not done and not truncated:
                avail_action = env.get_avail_actions()
                state = env.get_state()
                key, act_key = jax.random.split(key)
                actions = select_action(
                    actor,
                    jnp.asarray(obs),
                    jnp.asarray(avail_action).astype(jnp.bool_),
                    eps=epsilon,
                    act_key=act_key,
                )

                next_obs, reward, done, truncated, infos = env.step(actions)
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

            rb.add(episode)
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            if args.env_type == "smaclite":
                ep_stats.append(infos)
            num_episode += 1

        ## logging
        if len(ep_rewards) > args.log_every:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            writer.add_scalar("rollout/epsilon", epsilon, step)
            writer.add_scalar(
                "rollout/num_episodes", (training_step + 1) * args.batch_size, step
            )
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "rollout/battle_won",
                    np.mean([info["battle_won"] for info in ep_stats]),
                    step,
                )
            ep_rewards = []
            ep_lengths = []
            ep_stats = []

        ## Collate episodes in buffer into single batch
        batch = rb.get_batch()
        b_obs, b_avail_actions, b_actions, b_reward, b_states, b_done, b_mask = batch
        ### 1. Compute TD(λ) using "Reconciling λ-Returns with Experience Replay"(https://arxiv.org/pdf/1810.09967 Equation 3)
        ep_lens = []  # episode lengths
        for ep_idx in range(batch[2].shape[0]):
            ep_len = batch[-1][ep_idx].sum()
            ep_lens.append(ep_len.item())
        return_lambda = td_lambda(target_critic, batch, tuple(ep_lens), train_config)

        if args.normalize_return:
            ret_mu = return_lambda.mean(axis=-1)[b_mask].mean()
            ret_std = return_lambda.mean(axis=-1)[b_mask].std()
            return_lambda = (return_lambda - ret_mu) / ret_std

        ### 2. Update the critic
        critic, critic_optimizer, cr_loss, cr_gradients = critic_training_step(
            critic,
            critic_optimizer,
            batch=(b_states, b_obs, b_actions, b_mask),
            return_lambda=return_lambda,
        )
        training_step += 1

        if training_step % args.target_network_update_freq == 0:
            new_target_state = soft_update(
                nnx.state(target_critic), nnx.state(critic), args.polyak
            )
            nnx.update(target_critic, new_target_state)
        ### 3. Update actor

        actor, actor_optimizer, ac_loss, entropies, ac_gradients = actor_training_step(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            batch=(b_obs, b_avail_actions, b_states, b_actions, b_mask),
            train_config=train_config,
        )
        writer.add_scalar("train/critic_loss", cr_loss.item(), step)
        writer.add_scalar("train/actor_loss", ac_loss.item(), step)
        writer.add_scalar("train/entropy", entropies.item(), step)
        writer.add_scalar("train/actor_gradients", ac_gradients.item(), step)
        writer.add_scalar("train/critic_gradients", cr_gradients.item(), step)
        writer.add_scalar("train/num_updates", training_step, step)

        if training_step % args.eval_steps == 0:
            eval_obs, _ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            while eval_ep < args.num_eval_ep:
                key, act_key = jax.random.split(key)
                actions = select_action(
                    actor,
                    jnp.asarray(eval_obs),
                    jnp.asarray(eval_env.get_avail_actions()).astype(jnp.bool_),
                    eps=0,
                    act_key=act_key,
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
    env.close()
    eval_env.close()
