from multiprocessing import Pipe, Process
import torch
import tyro
import datetime
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import torch.nn.functional as F
from env.pettingzoo_wrapper import PettingZooWrapper
from env.smaclite_wrapper import SMACliteWrapper
from env.lbf import LBFWrapper
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    env_type: str = "smaclite"
    """ Pettingzoo, SMAClite ... """
    env_name: str = "3m"
    """ Name of the environment"""
    env_family: str ="mpe"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    batch_size: int = 3
    """ Number of episodes to collect in each rollout"""
    actor_hidden_dim: int = 32
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 64
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float =  0.0003
    """ Learning rate for the actor"""
    learning_rate_critic: float =  0.0003
    """ Learning rate for the critic"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    td_lambda: float = 0.9
    """ TD(λ) discount factor"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    normalize_advantage: bool = True
    """ Normalize the advantage if True"""
    normalize_return: bool = True
    """ Normalize the returns if True"""
    log_every: int = 10
    """ Logging steps """
    ppo_clip: float = 0.2
    """ PPO clipping factor """
    entropy_coef: float = 0.001
    """ Entropy coefficient """
    epochs: int = 3
    """ Number of training epochs"""
    clip_gradients: int = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    seed: int  = 1
    """ Random seed"""
    device: str ="cpu"
    """ Device (cpu, gpu, mps)"""
    eval_steps: int = 10
    """ Evaluate the policy each «eval_steps» training steps"""
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""


class  RolloutBuffer():
    def __init__(self,buffer_size,num_agents,obs_space,state_space,action_space,normalize_reward = False):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.normalize_reward = normalize_reward
        self.episodes = [None] * buffer_size
        self.pos = 0
    def add(self,episode):
        self.episodes[self.pos] = episode 
        self.pos += 1
    def get_batch(self):
        self.pos = 0
        lengths = [len(episode["obs"]) for episode in self.episodes ]
        max_length = max(lengths)
        obs = np.zeros((self.buffer_size,max_length,self.num_agents,self.obs_space))
        avail_actions = np.zeros((self.buffer_size,max_length,self.num_agents,self.action_space))
        actions = np.zeros((self.buffer_size,max_length,self.num_agents))
        log_probs = np.zeros((self.buffer_size,max_length,self.num_agents))
        reward = np.zeros((self.buffer_size,max_length))
        states = np.zeros((self.buffer_size,max_length,self.state_space))
        done = np.zeros((self.buffer_size,max_length))
        mask = torch.zeros(self.buffer_size, max_length,dtype=torch.bool)
        for i in range(self.buffer_size):
            length = lengths[i]
            obs[i,:length] = np.stack(self.episodes[i]["obs"])
            avail_actions[i,:length] = np.stack(self.episodes[i]["avail_actions"])
            actions[i,:length] = np.stack(self.episodes[i]["actions"])
            log_probs[i,:length] = np.stack(self.episodes[i]["log_prob"])
            reward[i,:length] = np.stack(torch.as_tensor(self.episodes[i]["reward"]))
            states[i,:length] = np.stack(self.episodes[i]["states"])
            done[i,:length] = np.stack(self.episodes[i]["done"])
            mask[i,:length] = 1
        if self.normalize_reward:
            mu = np.mean(reward[mask] )
            std = np.std(reward[mask] )
            reward[mask.bool()] = (reward[mask] - mu) /(std + 1e-6)
        self.episodes = [None] * self.buffer_size
        return (
            torch.from_numpy(obs).float(),
            torch.from_numpy(actions).long(),
            torch.from_numpy(log_probs).float(),
            torch.from_numpy(reward).float(),
            torch.from_numpy(states).float(),
            torch.from_numpy(avail_actions).bool(),
            torch.from_numpy(done).float(),
            mask,
        )
class Actor(nn.Module):
    def __init__(self, input_dim,hidden_dim,num_layer,output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))
        

    
    def act(self,x,avail_action=None):
        logits = self.logits(x,avail_action)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return action,distribution.log_prob(action)
    def logits(self,x,avail_action=None):
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, -1e9)
        return x

class Critic(nn.Module):
    def __init__(self, input_dim,hidden_dim,num_layer) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, 1)))
        

    
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def environment(env_type, env_name, env_family,agent_ids,kwargs):
    if env_type == 'pz':
        env = PettingZooWrapper(family = env_family, env_name = env_name,agent_ids=agent_ids,**kwargs)
    elif env_type == 'smaclite':
        env = SMACliteWrapper(map_name=env_name,agent_ids=agent_ids,**kwargs)
    elif env_type == 'lbf':
        env = LBFWrapper(map_name=env_name,agent_ids=agent_ids,**kwargs)
    
    return env
def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.tensor(norms), d)
    return total_norm_d
def soft_update(target_net, critic_net, polyak):
        print("did updated it ")
        for target_param, param in zip(target_net.parameters(), critic_net.parameters()):
            target_param.data.copy_(polyak * param.data + (1.0 - polyak) * target_param.data)

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

def env_worker(conn,env_serialized):
    env = env_serialized.env
    while True:
        task,content = conn.recv()
        if task == "reset":
            obs,_  = env.reset(seed=random.randint(0, 100000))
            avail_actions = env.get_avail_actions()
            state = env.get_state()
            content = {
                "obs": obs,
                "avail_actions": avail_actions,
                "state":state
            }
            conn.send(content)
        elif task == "get_env_info":
            content = {
                "obs_size":env.get_obs_size(),
                "action_size":env.get_action_size(),
                "n_agents": env.n_agents,
                "state_size": env.get_state_size()
            }
            conn.send(content)
        elif task == 'sample':
            actions = env.sample()
            content = {
                'actions':actions
            }
            conn.send(content)
        elif task == 'step':
            next_obs, reward, done, truncated, infos = env.step(content)
            avail_actions = env.get_avail_actions()
            state = env.get_state()
            content = {
                "next_obs":next_obs,
                "reward":reward,
                "done":done,
                "truncated":truncated,
                "infos":infos,
                "avail_actions":avail_actions,
                "next_state": state 
            }
            conn.send(content)
        elif task == "close":
            env.close()
            conn.close()
            break

if __name__ == "__main__":
    args = tyro.cli(Args)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ## import the environment 
    kwargs = {} #{"render_mode":'human',"shared_reward":False}
    ## Create the pipes to communicate between the main process (COMA algorithm) and child processes (envs)
    conns = [Pipe() for _ in range(args.batch_size)]
    ippo_conns, env_conns = zip(*conns)
    envs = [CloudpickleWrapper(environment(env_type= args.env_type,
                      env_name=args.env_name,
                      env_family=args.env_family,
                      agent_ids=args.agent_ids,
                      kwargs=kwargs)) for _ in range(args.batch_size)]
    processes = [Process(
        target=env_worker,
        args=(env_conns[i],envs[i]))
        for i in range(args.batch_size)]
    for process in processes:
            process.daemon = True
            process.start()
    eval_env = environment(env_type= args.env_type,
                      env_name=args.env_name,
                      env_family=args.env_family,
                      agent_ids=args.agent_ids,
                      kwargs=kwargs)
    
    ## Initialize the actor, critic and target-critic networks
    actor = Actor(
        input_dim=eval_env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=eval_env.get_action_size()
    )
    critic = Critic(input_dim=eval_env.get_obs_size(),
                    hidden_dim=args.critic_hidden_dim,
                    num_layer=args.critic_num_layers)
    Optimizer = getattr(optim, args.optimizer) 
    actor_optimizer = Optimizer(actor.parameters(),lr = args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(),lr = args.learning_rate_critic)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    writer = SummaryWriter(f"runs/IPPO-multienv-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    rb = RolloutBuffer( buffer_size=args.batch_size,
                        obs_space= eval_env.get_obs_size(),
                        state_space=eval_env.get_state_size(),
                        action_space=eval_env.get_action_size(),
                        num_agents= eval_env.n_agents,
                        normalize_reward= args.normalize_reward)
    step = 0
    training_step = 0
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    while step < args.total_timesteps:
        episodes = [{"obs": [],"actions":[],"log_prob":[],"reward":[],"states":[],"done":[],"avail_actions":[]}
                for _ in range(args.batch_size)]
        
        for ippo_conn in ippo_conns:
            ippo_conn.send(("reset",None))
        
        contents = [ippo_conn.recv() for ippo_conn in ippo_conns]
        obs  = np.stack([content["obs"] for content in contents],axis=0)
        avail_action = np.stack([content["avail_actions"] for content in contents],axis=0)
        state  = np.stack([content["state"] for content in contents])
        alive_envs = list(range(args.batch_size))      
        ep_reward, ep_length,ep_stat = [0]* args.batch_size,[0]* args.batch_size,[0]* args.batch_size
        while len(alive_envs) > 0:
            obs = torch.from_numpy(obs).to(args.device).float()
            avail_action = torch.tensor(avail_action, dtype=torch.bool, device=args.device)
            state = torch.from_numpy(state).to(args.device).float()
            with torch.no_grad():
                actions,log_probs = actor.act(obs,avail_action=avail_action)
            for i,j in enumerate(alive_envs):
                ippo_conns[j].send(("step",actions[i]))
            contents = [ippo_conns[i].recv() for i in alive_envs]
            next_obs =[content["next_obs"] for content in contents]
            reward = [content["reward"]   for content in contents]
            done = [content["done"]     for content in contents]
            truncated = [content["truncated"]     for content in contents]
            infos = [content.get("infos") for content in contents]
            next_avail_action = [content["avail_actions"] for content in contents]
            next_state = [content["next_state"]for content in contents]
            for i,j in enumerate(alive_envs):
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
            for i,j in enumerate(alive_envs[:]):
                if done[i] or truncated[i]:
                    alive_envs.remove(j)
                    rb.add(episodes[j])
                    episodes[j] = dict()
                    if args.env_type == 'smaclite':
                        ep_stat[j] = infos[i]
                else:
                    obs.append(next_obs[i]) 
                    avail_action.append(next_avail_action[i]) 
                    state.append(next_state[i])
            if obs:
                obs = np.stack(obs,axis=0)
                avail_action = np.stack(avail_action,axis=0)
                state = np.stack(state,axis=0)
        ep_rewards.extend(ep_reward)
        ep_lengths.extend(ep_length)
        if args.env_type == 'smaclite':
            ep_stats.extend([info["battle_won"] for info in ep_stat])
        ## logging
        if len(ep_rewards) > args.log_every:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length",np.mean(ep_lengths),step)
            if args.env_type == 'smaclite':
                writer.add_scalar("rollout/battle_won",np.mean(ep_stats), step)
            ep_rewards = []
            ep_lengths = []
            ep_stats   = []
        ## Collate episodes in buffer into single batch
        b_obs,b_actions,b_log_probs,b_reward,b_states,b_avail_actions,b_done,b_mask = rb.get_batch()
        return_lambda = torch.zeros_like(b_actions).float()
        advantages = torch.zeros_like(b_actions).float()
        with torch.no_grad():
            for ep_idx in range(return_lambda.size(0)):
                ep_len = b_mask[ep_idx].sum()
                last_return_lambda = 0
                for t in reversed(range(ep_len)):
                    if t == (ep_len -1):
                        next_value = 0
                    else:
                        next_value = critic(x = b_obs[ep_idx,t+1])
                    return_lambda[ep_idx,t] = last_return_lambda = b_reward[ep_idx,t] + args.gamma * (args.td_lambda * last_return_lambda + (1-args.td_lambda)*next_value)
                    advantages[ep_idx,t] = return_lambda[ep_idx,t] - critic(x = b_obs[ep_idx,t])
        # training loop
        if args.normalize_advantage:
            adv_mu = advantages.mean(dim=-1)[b_mask].mean()
            adv_std = advantages.mean(dim=-1)[b_mask].std()
            advantages = (advantages - adv_mu) / adv_std
        if args.normalize_return:
            ret_mu = return_lambda.mean(dim=-1)[b_mask].mean()
            ret_std = return_lambda.mean(dim=-1)[b_mask].std()
            return_lambda = (return_lambda - ret_mu) / ret_std
        actor_losses = []
        critic_losses = []
        entropies_bonuses =[]
        kl_divergences = []
        actor_gradients = []
        critic_gradients = []
        clipped_ratios = []
        for _ in range(args.epochs):
            actor_loss = 0
            critic_loss = 0
            entropies = 0
            kl_divergence = 0
            clipped_ratio = 0
            for t in range(b_obs.size(1)):
                # policy gradient (PG) loss
                ## PG: compute the ratio:
                current_logits = actor.logits(x= b_obs[:,t],avail_action=b_avail_actions[:,t])
                current_dist = Categorical(logits=current_logits)
                current_logprob = current_dist.log_prob(b_actions[:,t])
                log_ratio = current_logprob -b_log_probs[:,t]
                ratio = torch.exp(log_ratio)
                ## Compute PG the loss
                pg_loss1 = advantages[:,t] * ratio
                pg_loss2 = advantages[:,t] * torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip)
                pg_loss = torch.min(pg_loss1[b_mask[:,t]], pg_loss2[b_mask[:,t]]).mean()

                # Compute entropy bonus
                entropy_loss = current_dist.entropy()[b_mask[:,t]].mean()
                entropies += entropy_loss
                actor_loss += -pg_loss - args.entropy_coef*entropy_loss

                # Compute the value loss
                current_values = critic(x = b_obs[:,t])
                value_loss = F.mse_loss(current_values[b_mask[:,t]],return_lambda[:,t][b_mask[:,t]])
                critic_loss += value_loss

                # track kl distance
                b_kl_divergence = ((ratio -1) -log_ratio)[b_mask[:,t]].mean()
                kl_divergence+=b_kl_divergence
                clipped_ratio += ((ratio - 1.0).abs() > args.ppo_clip)[b_mask[:,t]].float().mean()
            actor_loss /= b_mask.sum()
            critic_loss /= b_mask.sum()
            entropies /= b_mask.sum()
            kl_divergence /= b_mask.sum()
            clipped_ratio /= b_mask.sum()
            
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            
            actor_loss.backward()
            critic_loss.backward()

            actor_gradient = norm_d([p.grad for p in actor.parameters() ],2)
            critic_gradient = norm_d([p.grad for p in critic.parameters() ],2)

            if args.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.clip_gradients)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.clip_gradients)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies_bonuses.append(entropies.item())
            kl_divergences.append(kl_divergence.item())
            actor_gradients.append(actor_gradient)
            critic_gradients.append(critic_gradient)
            clipped_ratios.append(clipped_ratio)

        writer.add_scalar("train/critic_loss", np.mean(critic_losses), step)
        writer.add_scalar("train/actor_loss", np.mean(actor_losses), step)
        writer.add_scalar("train/entropy", np.mean(entropies_bonuses), step)
        writer.add_scalar("train/kl_divergence", np.mean(kl_divergences), step)
        writer.add_scalar("train/clipped_ratios", np.mean(clipped_ratios), step)
        writer.add_scalar("train/actor_gradients", np.mean(actor_gradients) , step)
        writer.add_scalar("train/critic_gradients", np.mean(critic_gradients), step)
        training_step+= 1

        if training_step % args.eval_steps == 0:
            eval_obs,_ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            while eval_ep < args.num_eval_ep:
                eval_obs = torch.from_numpy(eval_obs).to(args.device).float()
                mask_eval = torch.tensor(eval_env.get_avail_actions(), dtype=torch.bool, device=args.device)
                with torch.no_grad():
                    actions,_ = actor.act(eval_obs,avail_action=mask_eval)
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
                    eval_ep +=1
            writer.add_scalar("eval/ep_reward",np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/std_ep_reward",np.std(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length",np.mean(eval_ep_length), step)
            if args.env_type == 'smaclite':
                writer.add_scalar("eval/battle_won",np.mean(np.mean([info["battle_won"] for info in eval_ep_stats])), step)
                



        
        




            




