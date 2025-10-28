from functools import partial
from typing import Tuple, Any
import jax
import optax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
from dataclasses import dataclass
import tyro
import random
import datetime
from env.pettingzoo_wrapper import PettingZooWrapper
from env.smaclite_wrapper import SMACliteWrapper
from env.lbf import LBFWrapper
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
    train_freq: int = 1
    """ Train the network each «train_freq» step in the environment"""
    optimizer: str = "adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0003
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0003
    """ Learning rate for the critic"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    target_network_update_freq: int = 1
    """ Update the target network each target_network_update_freq» step in the environment"""
    polyak: float = 0.005
    """ Polyak coefficient when using polyak averaging for target network update"""
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
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
    seed: int = 1
    """ Random seed"""


def gumbel_softmax(
    logits: jnp.ndarray,
    key: jax.Array,
    tau: float = 1.0,
    hard: bool = False,
    dim: int = -1,
):
    gumbels = jax.random.gumbel(key=key, shape=logits.shape)
    gumbels = (logits + gumbels) / tau
    y_soft = jax.nn.softmax(gumbels, axis=dim)
    if hard:
        index = y_soft.argmax(axis=dim, keepdims=True)
        y_hard = jax.numpy.zeros_like(logits)
        *batch, act_dim = y_hard.shape
        index = index.reshape(-1, index.shape[-1])
        y_hard = y_hard.reshape(-1, y_hard.shape[-1])
        y_hard = y_hard.at[jax.numpy.arange(y_hard.shape[0])[:, None], index].set(1)
        y_hard = y_hard.reshape(*batch, act_dim)
        ret = y_hard - jax.lax.stop_gradient(y_soft) + y_soft
    else:
        ret = y_soft
    return ret


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
        key: jax.Array,
        avail_action: jnp.ndarray | None = None,
        hard: bool = False,
    ):
        x = self.logits(x, avail_action)
        actions = gumbel_softmax(logits=x, key=key, hard=hard)
        return actions

    def logits(self, x: jnp.ndarray, avail_action: jnp.ndarray | None = None):
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = jnp.where(avail_action, x, -1e10)
        return x


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
        self.layers.append(nnx.Linear(hidden_dim, 1, rngs=rngs))

    def __call__(
        self,
        state: jnp.ndarray,
        actions: jnp.ndarray,
        grad_processing: bool = False,
        batch_action: jnp.ndarray | None = None,
    ):
        x = self.maddpg_inputs(state, actions, grad_processing, batch_action)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()

    def maddpg_inputs(
        self,
        state: jnp.ndarray,
        actions: jnp.ndarray,
        grad_processing: bool = False,
        batch_action: jnp.ndarray | None = None,
    ):
        maddpg_inputs = jnp.zeros((state.shape[0], self.num_agents, self.input_dim))
        maddpg_inputs = maddpg_inputs.at[:, :, : state.shape[-1]].set(
            jnp.expand_dims(state, axis=1)
        )
        oh = jnp.broadcast_to(
            array=jnp.expand_dims(actions, axis=1),
            shape=(
                actions.shape[0],
                self.num_agents,
                actions.shape[1],
                actions.shape[2],
            ),
        )
        oh = oh.reshape(actions.shape[0], self.num_agents, -1)
        if grad_processing:
            b_oh = jnp.broadcast_to(
                array=jnp.expand_dims(batch_action, axis=1),
                shape=(
                    batch_action.shape[0],
                    self.num_agents,
                    batch_action.shape[1],
                    batch_action.shape[2],
                ),
            )
            b_oh = b_oh.reshape(state.shape[0], self.num_agents, -1)
            mask = jnp.eye(self.num_agents)
            mask = jnp.broadcast_to(
                array=jnp.expand_dims(mask, axis=-1),
                shape=(mask.shape[0], mask.shape[1], batch_action.shape[2]),
            )
            mask = mask.reshape(self.num_agents, -1)
            oh = jnp.where(mask.astype(jnp.bool_), oh, b_oh)
        # maddpg_inputs = maddpg_inputs.at[:,:,state.shape[-1]:] = oh
        maddpg_inputs = maddpg_inputs.at[:, :, state.shape[-1] :].set(oh)
        return maddpg_inputs


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
        self.normalize_reward = normalize_reward
        self.rb_key = rb_key
        self.episodes = [None] * buffer_size
        self.pos = 0
        self.size = 0

    def store(self, episode: dict):
        self.episodes[self.pos] = episode
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int):
        self.rb_key, subkey = jax.random.split(self.rb_key)
        indices = jax.random.randint(subkey, (batch_size,), minval=0, maxval=self.size)
        batch = [self.episodes[i] for i in indices]
        lengths = [len(episode["obs"]) for episode in batch]
        max_length = max(lengths)
        obs = np.zeros((batch_size, max_length, self.num_agents, self.obs_space))
        avail_actions = np.zeros(
            (batch_size, max_length, self.num_agents, self.action_space)
        )
        actions = np.zeros((batch_size, max_length, self.num_agents, self.action_space))
        reward = np.zeros((batch_size, max_length))
        states = np.zeros((batch_size, max_length, self.state_space))
        done = np.ones((batch_size, max_length))
        mask = np.zeros((batch_size, max_length), dtype=np.bool_)
        if self.normalize_reward:
            rewards = []
        for i in range(batch_size):
            length = lengths[i]
            obs[i, :length] = np.stack(batch[i]["obs"])
            avail_actions[i, :length] = np.stack(batch[i]["avail_actions"])
            actions[i, :length] = np.stack(batch[i]["actions"])
            reward[i, :length] = np.stack(batch[i]["reward"])
            states[i, :length] = np.stack(batch[i]["states"])
            done[i, :length] = np.stack(batch[i]["done"])
            mask[i, :length] = 1
            if self.normalize_reward:
                rewards.extend(batch[i]["reward"])
        obs, avail_actions, actions, reward, states, done, mask = jax.tree.map(
            jnp.asarray, (obs, avail_actions, actions, reward, states, done, mask)
        )
        if self.normalize_reward:
            mu = np.mean(rewards)
            std = np.std(rewards)
            reward = (reward - mu) / (std + 1e-6)
        return (obs, actions, reward, states, avail_actions, done, mask)


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
    actor: nnx.Module,
    obs: jnp.ndarray,
    key: jax.Array,
    avail_action: jnp.ndarray | None = None,
):
    actions = actor(
        obs, key, avail_action=avail_action, hard=True
    )  ## These are one hot-vectors
    actions_to_take = actions.argmax(axis=-1)
    return actions, actions_to_take


def critic_loss(
    critic: nnx.Module,
    target_critic: nnx.Module,
    target_actor: nnx.Module,
    batch: Tuple[jnp.ndarray],
    train_key: jax.Array,
    gamma: float,
):
    (
        batch_obs,
        batch_action,
        batch_reward,
        batch_states,
        batch_avail_action,
        batch_done,
        batch_mask,
    ) = jax.tree.map(lambda x: jnp.moveaxis(x, 1, 0), batch)

    def q_value_t(carry, batch_t):
        q_values = critic(batch_t[0], batch_t[1])
        return carry, q_values

    _, q_values = jax.lax.scan(f=q_value_t, init=0, xs=(batch_states, batch_action))
    targets = jnp.zeros_like(q_values)

    def target_t(carry, batch_t):
        carry, touse_key = jax.random.split(carry)
        obs_t, avail_action_t, states_t, reward_t, done_t = batch_t
        actions_from_target_actor = target_actor(
            obs_t, touse_key, avail_action=avail_action_t, hard=True
        )
        qvals_from_taget_critic = target_critic(states_t, actions_from_target_actor)
        # qvals_from_taget_critic = jnp.where(qvals_from_taget_critic==jnp.nan,0.0,qvals_from_taget_critic)
        reward_t, done_t = jax.tree.map(
            lambda x: jnp.broadcast_to(
                array=jnp.expand_dims(x, axis=-1), shape=(x.shape[0], obs_t.shape[-2])
            ),
            (reward_t, done_t),
        )
        targets = reward_t + gamma * (1 - done_t) * qvals_from_taget_critic
        return carry, targets

    train_key, trunc_targets = jax.lax.scan(
        f=target_t,
        init=train_key,
        xs=(
            batch_obs[1:],
            batch_avail_action[1:],
            batch_states[1:],
            batch_reward[:-1],
            batch_done[:-1],
        ),
    )
    targets = targets.at[:-1].set(trunc_targets)
    targets = targets.at[-1].set(jnp.expand_dims(batch_reward[-1], axis=-1))
    loss = optax.l2_loss(jax.lax.stop_gradient(targets), q_values)

    loss = jnp.where(batch_mask, loss.mean(axis=-1), 0).sum()
    return loss / batch_mask.sum(), train_key


@partial(jax.jit, static_argnums=6)
def critic_training_step(
    critic: nnx.Module,
    target_critic: nnx.Module,
    target_actor: nnx.Module,
    critic_optimizer: nnx.Optimizer,
    batch: Tuple[jnp.ndarray],
    train_key: jax.Array,
    gamma: float,
):
    (loss, train_key), grads = nnx.value_and_grad(critic_loss, has_aux=True)(
        critic, target_critic, target_actor, batch, train_key, gamma
    )
    g_norm = optax.global_norm(grads)
    critic_optimizer.update(critic, grads)
    return critic, critic_optimizer, loss, g_norm, train_key


def actor_loss(
    actor: nnx.Module,
    critic: nnx.Module,
    batch: Tuple[jnp.ndarray],
    train_key: jax.Array,
):
    def actor_loss_t(carry, batch_t):
        prev_loss, prev_key = carry
        obs_t, action_t, avail_action_t, states_t, mask_t = batch_t
        prev_key, touse_key = jax.random.split(prev_key)
        actions = actor(obs_t, touse_key, avail_action=avail_action_t, hard=False)
        qvals = critic(states_t, actions, grad_processing=True, batch_action=action_t)
        ac_loss = jnp.where(mask_t, qvals.sum(axis=-1), 0).sum()
        carry = prev_loss - ac_loss
        return (carry, prev_key), None

    (ac_loss, train_key), _ = jax.lax.scan(
        f=actor_loss_t,
        init=(0, train_key),
        xs=jax.tree.map(lambda x: jnp.moveaxis(x, 1, 0), batch),
    )
    return ac_loss / batch[-1].sum(), train_key


@nnx.jit
def actor_training_step(
    actor: nnx.Module,
    critic: nnx.Module,
    actor_optimizer: nnx.Optimizer,
    batch: Tuple[jnp.ndarray],
    train_key: jax.Array,
):
    (loss, train_key), grads = nnx.value_and_grad(actor_loss, has_aux=True)(
        actor, critic, batch, train_key
    )
    g_norm = optax.global_norm(grads)
    actor_optimizer.update(actor, grads)
    return actor, actor_optimizer, loss, g_norm, train_key


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Set the random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.key(seed)
    key, rb_key, train_key = jax.random.split(key, num=3)
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

    actor = Actor(
        input_dim=env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=env.get_action_size(),
        rngs=rngs,
    )
    target_actor = nnx.clone(actor)

    maddpg_input_dim = env.get_state_size() + env.n_agents * env.get_action_size()
    critic = Critic(
        input_dim=maddpg_input_dim,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        output_dim=env.get_action_size(),
        num_agents=env.n_agents,
        rngs=rngs,
    )
    target_critic = nnx.clone(critic)
    # Optimizer
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
            name=f"MADDPG-JAX-{run_name}",
        )
    writer = SummaryWriter(f"runs/MADDPG-JAX-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space=env.get_obs_size(),
        state_space=env.get_state_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        rb_key=rb_key,
        normalize_reward=args.normalize_reward,
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
        obs, _ = env.reset()
        ep_reward, ep_length = 0, 0
        done, truncated = False, False
        while not done and not truncated:
            avail_action = env.get_avail_actions()
            state = env.get_state()
            key, act_key = jax.random.split(key)
            actions, actions_to_take = select_action(
                actor,
                jnp.asarray(obs),
                act_key,
                jnp.asarray(avail_action).astype(jnp.bool_),
            )

            next_obs, reward, done, truncated, infos = env.step(actions_to_take)
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

        rb.store(episode)
        num_episode += 1
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)
        if args.env_type == "smaclite":
            ep_stats.append(infos)  ## Add battle won for smaclite

        if num_episode % args.log_every == 0:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
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
        if num_episode > args.batch_size:
            if num_episode % args.train_freq == 0:
                batch = rb.sample(args.batch_size)
                ## train the critic
                critic, critic_optimizer, cr_loss, critic_gradients, train_key = (
                    critic_training_step(
                        critic,
                        target_critic,
                        target_actor,
                        critic_optimizer,
                        batch,
                        train_key,
                        args.gamma,
                    )
                )
                ## train the actor
                actor, actor_optimizer, ac_loss, actor_gradients, train_key = (
                    actor_training_step(
                        actor,
                        critic,
                        actor_optimizer,
                        batch=(batch[0], batch[1], batch[4], batch[3], batch[-1]),
                        train_key=train_key,
                    )
                )
                num_updates += 1
                writer.add_scalar("train/critic_loss", cr_loss.item(), step)
                writer.add_scalar("train/actor_loss", ac_loss.item(), step)
                writer.add_scalar("train/actor_gradients", actor_gradients.item(), step)
                writer.add_scalar(
                    "train/critic_gradients", critic_gradients.item(), step
                )
                writer.add_scalar("train/num_updates", num_updates, step)

            if num_episode % args.target_network_update_freq == 0:
                new_target_state = soft_update(
                    nnx.state(target_actor), nnx.state(actor), args.polyak
                )
                nnx.update(target_actor, new_target_state)

                new_target_state = soft_update(
                    nnx.state(target_critic), nnx.state(critic), args.polyak
                )
                nnx.update(target_critic, new_target_state)

        if num_episode % args.eval_steps == 0:
            eval_obs, _ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            while eval_ep < args.num_eval_ep:
                key, act_key = jax.random.split(key)
                _, eval_actions = select_action(
                    actor,
                    jnp.asarray(eval_obs),
                    act_key,
                    jnp.asarray(eval_env.get_avail_actions()).astype(jnp.bool_),
                )
                next_obs_, reward, done, truncated, infos = eval_env.step(eval_actions)
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
