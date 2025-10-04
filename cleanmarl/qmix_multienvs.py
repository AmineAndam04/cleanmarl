from multiprocessing import Pipe, Process
import copy
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
    env_type: str = "smaclite" #"pz"
    """ Pettingzoo, SMAClite ... """
    env_name: str = "MMM" #"simple_spread_v3" #"pursuit_v4"
    """ Name of the environment """
    env_family: str ="mpe"
    """ Env family when using pz"""
    num_envs: int  = 4
    """ Number of parallel environments"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    buffer_size: int = 5000
    """ The number of episodes in the replay buffer"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    train_freq: int = 2
    """ Train the network each «train_freq» step in the environment"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate: float =  0.0005
    """ Learning rate"""
    batch_size: int = 32
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
    polyak: float = 0.005
    """ Polyak coefficient when using polyak averaging for target network update"""
    clip_gradients: int = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    log_every: int = 10
    """ Logging steps """
    eval_steps: int = 50
    """ Evaluate the policy each «eval_steps» steps"""
    num_eval_ep: int = 5
    """ Number of evaluation episodes"""
    device: str ="cpu"
    """ Device (cpu, gpu, mps)"""
    seed: int = 42
    """ Random seed""" 
    n_epochs: int = 2
    """ Number of batches sampled in one update"""
    seed: int  = 1
    """ Random seed"""

    

class Qnetwrok(nn.Module):
    def __init__(self, input_dim,hidden_dim,num_layer,output_dim) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))
    def forward(self,x,avail_action=None):
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, float('-inf'))
        return x

class MixingNetwork(nn.Module):
    def __init__(self, n_agents,s_dim,hidden_dim):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.hypernet_weight_1 = nn.Linear(s_dim, n_agents*hidden_dim)
        self.hypernet_bias_1 = nn.Linear(s_dim, hidden_dim)
        self.hypernet_weight_2 = nn.Linear(s_dim, hidden_dim)
        self.hypernet_bias_2 = nn.Sequential(nn.Linear(s_dim, hidden_dim),
                                             nn.ReLU(),
                                            nn.Linear(hidden_dim, 1))
    def forward(self,Q,s):
        Q = Q.reshape(-1,1,self.n_agents)
        W1 = torch.abs(self.hypernet_weight_1(s))
        W1 = W1.reshape(-1,self.n_agents ,self.hidden_dim)
        b1 = self.hypernet_bias_1(s)
        b1 = b1.reshape(-1,1,self.hidden_dim)
        Q = nn.functional.elu(torch.bmm(Q, W1) + b1)

        W2 = torch.abs(self.hypernet_weight_2(s))
        W2 = W2.reshape(-1,self.hidden_dim,1)
        b2 = self.hypernet_bias_2(s)
        b2 = b2.reshape(-1,1,1)
        Q_tot = torch.bmm(Q, W2) + b2
        return Q_tot
class ReplayBuffer:
    def __init__(self,buffer_size,num_agents,obs_space,state_space,action_space,normalize_reward = False):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.normalize_reward = normalize_reward

        self.episodes = [None] * buffer_size
        self.pos = 0
        self.size = 0
    def store(self,episode):
        self.episodes[self.pos] = episode #(obs,action,reward,done,next_obs,mask)
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    def sample(self,batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = [self.episodes[i] for i in indices]
        lengths = [len(episode["obs"]) for episode in batch ]
        max_length = max(lengths)
        obs = np.zeros((batch_size,max_length,self.num_agents,self.obs_space))
        avail_actions = np.zeros((batch_size,max_length,self.num_agents,self.action_space))
        actions = np.zeros((batch_size,max_length,self.num_agents))
        reward = np.zeros((batch_size,max_length))
        next_obs = np.zeros((batch_size,max_length,self.num_agents,self.obs_space))
        states = np.zeros((batch_size,max_length,self.state_space))
        next_states = np.zeros((batch_size,max_length,self.state_space))
        done = np.zeros((batch_size,max_length))
        mask = torch.zeros(batch_size, max_length,dtype=torch.bool)
        for i in range(batch_size):
            length = lengths[i]
            obs[i,:length] =np.stack(batch[i]["obs"])
            avail_actions[i,:length] =np.stack(batch[i]["avail_actions"])
            actions[i,:length] =np.stack(batch[i]["actions"])
            reward[i,:length] =np.stack(batch[i]["reward"])
            next_obs[i,:length] =np.stack(batch[i]["next_obs"])
            states[i,:length] =np.stack(batch[i]["states"])
            next_states[i,:length] =np.stack(batch[i]["next_states"])
            done[i,:length] =np.stack(batch[i]["done"])
            mask[i,:length] = 1

        if self.normalize_reward:
            mu = np.mean(reward[mask] )
            std = np.std(reward[mask] )
            reward[mask.bool()] = (reward[mask] - mu) /(std + 1e-6)
        
        return (
            torch.from_numpy(obs).float(),
            torch.from_numpy(actions).long(),
            torch.from_numpy(reward).float(),
            torch.from_numpy(next_obs).float(),
            torch.from_numpy(states).float(),
            torch.from_numpy(next_states).float(),
            torch.from_numpy(avail_actions).bool(),
            torch.from_numpy(done).float(),
            mask,
        )

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
def soft_update(target_net, utility_net, polyak):
        for target_param, param in zip(target_net.parameters(), utility_net.parameters()):
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
    # Set the seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    kwargs = {} #{"render_mode":'human',"shared_reward":False}

    ## Create the pipes to communicate between the main process (QMIX algorithm) and child processes (envs)
    conns = [Pipe() for _ in range(args.num_envs)]
    qmix_conns, env_conns = zip(*conns)
    envs = [CloudpickleWrapper(environment(env_type= args.env_type,
                      env_name=args.env_name,
                      env_family=args.env_family,
                      agent_ids=args.agent_ids,
                      kwargs=kwargs)) for _ in range(args.num_envs)]
    processes = [Process(
        target=env_worker,
        args=(env_conns[i],envs[i]))
        for i in range(args.num_envs)]
    for process in processes:
            process.daemon = True
            process.start()
    
    eval_env = environment(env_type= args.env_type,
                      env_name=args.env_name,
                      env_family=args.env_family,
                      agent_ids=args.agent_ids,
                      kwargs=kwargs)
    ## initialize the utility and target networks
    ## initialize the utility and target networks
    utility_network = Qnetwrok(input_dim=eval_env.get_obs_size(),
                                          hidden_dim=args.hidden_dim,
                                          num_layer=args.num_layers,
                                          output_dim=eval_env.get_action_size())
    target_network = copy.deepcopy(utility_network)
    mixer = MixingNetwork(n_agents=eval_env.n_agents,
                                    s_dim=eval_env.get_state_size(),
                                    hidden_dim=args.hyper_dim)
    target_mixer =copy.deepcopy(mixer)

    
    ## initialize the optimizer
    optimizer = getattr(optim, args.optimizer) 
    optimizer = optimizer(list(utility_network.parameters()) + list(mixer.parameters()),
                          lr = args.learning_rate)

    ## initialize a shared replay buffer
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space= eval_env.get_obs_size(),
        state_space=eval_env.get_state_size(),
        action_space=eval_env.get_action_size(),
        num_agents= eval_env.n_agents,
        normalize_reward= args.normalize_reward)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    writer = SummaryWriter(f"runs/QMIX-multienvs-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    num_episodes = 0
    num_updates = 0
    step = 0
    while step < args.total_timesteps:
        episodes = [{"obs": [],"actions":[],"reward":[],"next_obs":[],"states":[], "next_states":[],"done":[],"avail_actions":[]}
                   for _ in range(args.num_envs)]

        for qmix_conn in qmix_conns:
            qmix_conn.send(("reset",None))
        contents = [qmix_conn.recv() for qmix_conn in qmix_conns]
        obs  = np.stack([content["obs"] for content in contents],axis=0)
        avail_action = np.stack([content["avail_actions"] for content in contents],axis=0)
        state  = [content["state"] for content in contents]
        alive_envs = list(range(args.num_envs))

        ep_reward, ep_length = [0]* args.num_envs,[0]* args.num_envs
        
        while len(alive_envs) > 0:
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, step)
            if random.random() < epsilon:
                for i in alive_envs:
                    qmix_conns[i].send(("sample",None))
                contents = [qmix_conns[i].recv() for i in alive_envs]
                actions = np.array([content["actions"] for content in contents])
            else:
                obs = torch.from_numpy(obs).to(args.device).float()
                avail_action = torch.tensor(avail_action, dtype=torch.bool, device=args.device)
                num_alive_envs =len(alive_envs)
                with torch.no_grad():
                    q_values = utility_network(obs,avail_action =avail_action)
                actions  = torch.argmax(q_values,dim=-1)
            # Send actions
            for i,j in enumerate(alive_envs):
                    qmix_conns[j].send(("step",actions[i]))
            contents = [qmix_conns[i].recv() for i in alive_envs]
            next_obs =[content["next_obs"] for content in contents]
            reward = [content["reward"]   for content in contents]
            done = [content["done"]     for content in contents]
            truncated = [content["truncated"]     for content in contents]
            avail_action = [content["avail_actions"] for content in contents]
            infos = [content.get("infos") for content in contents]
            next_state = [content["next_state"]for content in contents]
            
            for i,j in enumerate(alive_envs):
                episodes[j]["obs"].append(obs[i])
                episodes[j]["actions"].append(actions[i])
                episodes[j]["reward"].append(reward[i])
                episodes[j]["next_obs"].append(next_obs[i])
                episodes[j]["done"].append(done[i])
                episodes[j]["avail_actions"].append(avail_action[i])
                episodes[j]["states"].append(state[i])
                episodes[j]["next_states"].append(next_state[i])
                ep_reward[j] += reward[i]
                ep_length[j] += 1
            
            step += len(alive_envs)

            obs = []
            state = []
            temp_avail_action = []

            for i,j in enumerate(alive_envs[:]):
                if done[i] or truncated[i]:
                    alive_envs.remove(j)
                    rb.store(episodes[j])
                    episodes[j] = dict()
                    if args.env_type == 'smaclite':## Add battle won for smaclite
                        ep_stats.append(infos[i]) 
                else:
                    obs.append(next_obs[i]) 
                    temp_avail_action.append(avail_action[i]) 
                    state.append(next_state[i])
            if obs:
                obs = np.stack(obs,axis=0)
                avail_action = np.stack(temp_avail_action,axis=0)
            
        
        num_episodes += args.num_envs
        ep_rewards.extend(ep_reward)
        ep_lengths.extend(ep_length)
        

        if num_episodes > args.batch_size:
            if num_episodes % args.train_freq == 0:
                losses = 0
                gradients = 0
                for _ in range(args.n_epochs):
                    batch_obs,batch_action,batch_reward,batch_next_obs,batch_states,batch_next_states,batch_avail_action,batch_done, batch_mask = rb.sample(args.batch_size)
                    loss = 0
                    for t in range(batch_obs.size(1)):
                        with torch.no_grad():
                            q_next_max,_ = target_network(batch_next_obs[:,t],avail_action =batch_avail_action[:,t] ).max(dim=-1)
                            q_tot_target = target_mixer(Q = q_next_max, s = batch_next_states[:,t]).squeeze()
                            targets = batch_reward[:,t] + args.gamma * (1-batch_done[:,t])*q_tot_target #(torch.sum(q_next_max, dim=-1))
                        
                        q_values = torch.gather(utility_network(batch_obs[:,t]),dim=-1, index=batch_action[:,t].unsqueeze(-1)).squeeze()
                        q_tot = mixer(Q = q_values,s= batch_states[:,t]).squeeze() 
                        loss += F.mse_loss(targets[batch_mask[:,t]],q_tot[batch_mask[:,t]]) * batch_mask[:,t].sum()
                    loss /= batch_mask.sum()
                    optimizer.zero_grad()
                    loss.backward()
                    num_updates += 1
                    grads = [p.grad for p in list(utility_network.parameters()) + list(mixer.parameters()) ]
                    qmix_gradient = norm_d(grads,2)
                    if args.clip_gradients > 0:
                        torch.nn.utils.clip_grad_norm_(
                            list(utility_network.parameters()) + list(mixer.parameters()),args.clip_gradients)
                    optimizer.step()

                    losses += loss.item()
                    gradients += qmix_gradient 
                
                losses /= args.n_epochs
                gradients /= args.n_epochs
                writer.add_scalar("train/loss", losses, step)
                writer.add_scalar("train/grads", gradients, step)
                writer.add_scalar("train/num_updates", num_updates, step)
                
            if (num_updates//args.n_epochs) % args.target_network_update_freq == 0:
                    soft_update(
                        target_net=target_network,
                        utility_net=utility_network,
                        polyak=args.polyak
                    )
                    soft_update(
                        target_net=target_mixer,
                        utility_net=mixer,
                        polyak=args.polyak
                    )
        if (num_updates//args.n_epochs) %  args.log_every == 0:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length",np.mean(ep_lengths),step)
            writer.add_scalar("rollout/epsilon",epsilon,step)
            writer.add_scalar("rollout/num_episodes",num_episodes,step)
            if args.env_type == 'smaclite':
                writer.add_scalar("rollout/battle_won",np.mean([info["battle_won"] for info in ep_stats]), step)
            ep_rewards = []
            ep_lengths = []
            ep_stats   = []
        

        if (num_updates//args.n_epochs) %  args.eval_steps == 0:
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
                q_values = utility_network(eval_obs, avail_action = mask_eval)
                actions  = torch.argmax(q_values,dim=-1)
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

    for conn in qmix_conns:
        conn.send(("close", None))
    for p in processes:
        p.join()
    writer.close()