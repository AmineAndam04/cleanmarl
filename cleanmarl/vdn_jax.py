from typing import Tuple, Any
import jax
import optax
from flax import nnx
import jax.numpy as jnp
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
    env_family: str = "sisl"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    buffer_size: int = 10000
    """ The size of the replay buffer"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    learning_starts: int = 5000
    """ Number of env steps to initialize the replay buffer"""
    train_freq: int = 5
    """ Train the network each «train_freq» step in the environment"""
    optimizer: str = "adam"
    """ The optimizer"""
    learning_rate: float = 0.0005
    """ Learning rate"""
    batch_size: int = 32
    """ Batch size"""
    start_e: float = 1
    """ The starting value of epsilon, for exploration"""
    end_e: float = 0.05
    """ The end value of epsilon, for exploration"""
    exploration_fraction: float = 0.05
    """ The fraction of «total-timesteps» it takes from to go from start_e to  end_e"""
    hidden_dim: int = 64
    """ Hidden dimension"""
    num_layers: int = 1
    """ Number of layers"""
    target_network_update_freq: int = 5
    """ Update the target network each target_network_update_freq» step in the environment"""
    polyak: float = 0.005
    """ Polyak coefficient when using polyak averaging for target network update"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    clip_gradients: float = 5
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    log_every: int = 10
    """ Log rollout stats every <log_every> episode """
    eval_steps: int = 5000
    """ Evaluate the policy each «eval_steps» steps"""
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


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_space: int,
        action_space: int,
        rb_key: jax.Array,
        normalize_reward: bool = False,
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.normalize_reward = normalize_reward
        self.rb_key = rb_key
        self.obs = np.zeros(
            (self.buffer_size, self.num_agents, self.obs_space), dtype=np.float32
        )
        self.action = np.zeros((self.buffer_size, self.num_agents), dtype=np.int32)
        self.reward = np.zeros((self.buffer_size), dtype=np.float32)
        self.next_obs = np.zeros(
            (self.buffer_size, self.num_agents, self.obs_space), dtype=np.float32
        )
        self.next_avail_action = np.zeros(
            (self.buffer_size, self.num_agents, self.action_space), dtype=np.bool_
        )
        self.done = np.zeros((self.buffer_size), dtype=np.int32)
        self.pos = 0
        self.size = 0

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_obs: np.ndarray,
        next_avail_action: np.ndarray,
    ):
        self.obs[self.pos] = obs
        self.action[self.pos] = action
        self.reward[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.next_avail_action[self.pos] = next_avail_action
        self.done[self.pos] = done
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

        return obs, action, reward, next_obs, next_avail, done


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
    gamma: float,
):
    (
        batch_obs,
        batch_action,
        batch_reward,
        batch_next_obs,
        batch_next_avail_action,
        batch_done,
    ) = batch
    q_next_max = target_network(
        batch_next_obs, avail_action=batch_next_avail_action
    ).max(axis=-1)
    vdn_q_max = q_next_max.sum(axis=-1)
    targets = batch_reward + gamma * (1 - batch_done) * vdn_q_max
    q_values = jnp.take_along_axis(
        arr=utility_network(batch_obs),
        indices=jnp.expand_dims(batch_action, axis=-1),
        axis=-1,
    ).squeeze()
    vdn_q_values = q_values.sum(axis=-1)
    loss = optax.l2_loss(jax.lax.stop_gradient(targets), vdn_q_values).mean()
    return loss


@nnx.jit
def training_step(
    utility_network: nnx.Module,
    target_network: nnx.Module,
    optimizer: nnx.Optimizer,
    batch: Tuple[jnp.ndarray],
    gamma: float,
):
    loss, grads = nnx.value_and_grad(loss_fn)(
        utility_network, target_network, batch, gamma
    )
    g_norm = optax.global_norm(grads)
    optimizer.update(utility_network, grads)
    return utility_network, optimizer, loss, g_norm


@nnx.jit
def select_action(network: Qnetwork, obs: jnp.ndarray, avail_action: jnp.ndarray):
    q_values = network(obs, avail_action)
    return jnp.argmax(q_values, axis=-1)


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Set the randomness
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.key(seed)
    key, rb_key = jax.random.split(key)
    rngs = nnx.Rngs(seed)
    # Import the environment
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
    # Initialize the utility and target networks
    utility_network = Qnetwork(
        input_dim=env.get_obs_size(),
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layers,
        output_dim=env.get_action_size(),
        rngs=rngs,
    )
    target_network = nnx.clone(utility_network)

    # Initialize the optimizer
    optimizer = getattr(optax, args.optimizer)(learning_rate=args.learning_rate)
    if args.clip_gradients > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.clip_gradients), optimizer
        )
    optimizer = nnx.Optimizer(utility_network, optimizer, wrt=nnx.Param)
    # Initialize a shared replay buffer
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space=env.get_obs_size(),
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
            name=f"VDN-JAX-{run_name}",
        )
    writer = SummaryWriter(f"runs/VDN-JAX-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    obs, _ = env.reset()
    avail_action = env.get_avail_actions()
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    ep_reward = 0
    ep_length = 0
    num_episodes = 0
    num_updates = 0
    for step in range(args.total_timesteps):
        ## select actions
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
                utility_network,
                jnp.asarray(obs),
                jnp.asarray(avail_action).astype(jnp.bool_),
            )

        next_obs, reward, done, truncated, infos = env.step(actions)
        next_avail_action = env.get_avail_actions()  # We need the next_avail_action to compute the target loss : max of Q(next_state)

        ep_reward += reward
        ep_length += 1

        rb.store(obs, np.asarray(actions), reward, done, next_obs, next_avail_action)
        obs = next_obs
        avail_action = next_avail_action
        if done or truncated:
            obs, _ = env.reset()
            avail_action = env.get_avail_actions()
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            if args.env_type == "smaclite":
                ep_stats.append(infos)  ## Add battle won for smaclite
            ep_reward = 0
            ep_length = 0
            num_episodes += 1

        if step > args.learning_starts:
            if step % args.train_freq == 0:
                batch = rb.sample(args.batch_size)
                utility_network, optimizer, loss, g_norm = training_step(
                    utility_network, target_network, optimizer, batch, args.gamma
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

        if step > 0 and step % args.eval_steps == 0:
            eval_obs, _ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            while eval_ep < args.num_eval_ep:
                actions = select_action(
                    utility_network,
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
