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
    env_name: str = "3m" #"simple_spread_v3" #"pursuit_v4"
    """ Name of the environment """
    env_family: str = "mpe" #"sisl"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    num_envs: int  = 4
    """ Number of parallel environments"""
    buffer_size: int = 10000
    """ The size of the replay buffer"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    learning_starts: int = 5000 
    """ Number of env steps to initialize the replay buffer"""
    train_freq: int = 2 
    """ Train the network each «train_freq» step in the environment. The used value is train_freq*num_envs"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate: float =  0.0005
    """ Learning rate"""
    batch_size: int = 16
    """ Batch size, the actual batch_size is (batch_size*num_envs)"""
    clip_gradients: int = 5
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
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
    target_network_update_freq: int = 1
    """ Frequency of updating target network. The used value is target_network_update_freq*num_envs"""
    polyak: float = 0.005
    """ Update the target network each target_network_update_freq» step in the environment"""
    log_every: int = 10
    """ Logging steps"""
    normalize_reward: bool = False
    """ Normalize the rewards"""
    eval_steps: int = 5000
    """ Evaluate the policy each eval_steps steps. The used value is eval_steps*num_envs"""
    num_eval_ep: int = 5
    """ Number of evaluation episodes"""
    device: str ="mps"
    """ Device (cpu, gpu, mps)"""
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
class ReplayBuffer:
    def __init__(self,buffer_size,num_agents,obs_space,action_space,num_envs,normalize_reward= False):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_envs = num_envs
        self.normalize_reward = normalize_reward

        self.obs = np.zeros((self.buffer_size,self.num_envs,self.num_agents,self.obs_space))
        self.action = np.zeros((self.buffer_size,self.num_envs,self.num_agents))
        self.reward = np.zeros((self.buffer_size,self.num_envs))
        self.next_obs = np.zeros((self.buffer_size,self.num_envs,self.num_agents,self.obs_space))
        self.next_avail_action = np.zeros((self.buffer_size,self.num_envs,self.num_agents,self.action_space))
        self.done = np.zeros((self.buffer_size,self.num_envs))
        self.pos = 0
        self.size = 0
    def store(self,obs,action,reward,done,next_obs,next_avail_action):
        self.obs[self.pos] = obs
        self.action[self.pos] = action
        self.reward[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.next_avail_action[self.pos]  = next_avail_action
        self.done[self.pos] = done
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    def sample(self,batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        if self.normalize_reward:
            mu = np.mean(self.reward[indices])
            std = np.std(self.reward[indices])
            rewards = (self.reward[indices] - mu) /(std + 1e-6)
        else :
            rewards  = self.reward[indices]
        return (
            torch.from_numpy(self.obs[indices]).reshape(batch_size*self.num_envs,self.num_agents,self.obs_space).float(),
            torch.from_numpy(self.action[indices]).reshape(batch_size*self.num_envs,self.num_agents).long(),
            torch.from_numpy(rewards).reshape(batch_size*self.num_envs).float(),
            torch.from_numpy(self.next_obs[indices]).reshape(batch_size*self.num_envs,self.num_agents,self.obs_space).float(),
            torch.from_numpy(self.next_avail_action[indices]).reshape(batch_size*self.num_envs,self.num_agents,self.action_space).bool(),
            torch.from_numpy(self.done[indices]).reshape(batch_size*self.num_envs).float()
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
            content = {
                "obs": obs,
                "avail_actions": avail_actions
            }
            conn.send(content)
        elif task == "get_env_info":
            content = {
                "obs_size":env.get_obs_size(),
                "action_size":env.get_action_size(),
                "n_agents": env.n_agents
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
            content = {
                "next_obs":next_obs,
                "reward":reward,
                "done":done,
                "truncated":truncated,
                "infos":infos,
                "avail_actions":avail_actions,
            }
            conn.send(content)
        elif task == "close":
            env.close()
            conn.close()
            break
        

if __name__ == "__main__":
    
    args = tyro.cli(Args)
    #Set the seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    kwargs = {} #{"render_mode":'human',"shared_reward":False}
    ## Create the pipes to communicate between the main process (VDN algorithm) and child processes (envs)
    conns = [Pipe() for _ in range(args.num_envs)]
    vdn_conns, env_conns = zip(*conns)
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
    
    # Initialize the netowrks
    utility_network = Qnetwrok(input_dim=eval_env.get_obs_size(),
                                          hidden_dim=args.hidden_dim,
                                          num_layer=args.num_layers,
                                          output_dim=eval_env.get_action_size())
    target_network = copy.deepcopy(utility_network)

    # Initialize the optimizer
    optimizer = getattr(optim, args.optimizer) 
    optimizer = optimizer(utility_network.parameters(),lr = args.learning_rate)

    ## initialize a shared replay buffer
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space= eval_env.get_obs_size(),
        action_space=eval_env.get_action_size(),
        num_agents= eval_env.n_agents,
        num_envs = args.num_envs,
        normalize_reward= args.normalize_reward
    )

    
    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    writer = SummaryWriter(f"runs/VDN-multienvs-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Reset the environments
    for vdn_conn in vdn_conns:
        vdn_conn.send(("reset",None))
    contents = [vdn_conn.recv() for vdn_conn in vdn_conns]
    obs  = np.stack([content["obs"]            for content in contents], axis=0)
    avail_action = np.stack([content["avail_actions"] for content in contents], axis=0)

    ep_reward = np.zeros(args.num_envs)
    ep_length = np.zeros(args.num_envs)
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    step = 0
    num_updates = 0
    while step < args.total_timesteps:
        
        ## select actions
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, step)
        if random.random() < epsilon:
            actions = []
            for vdn_conn in vdn_conns:
                vdn_conn.send(("sample",None))
            contents = [vdn_conn.recv() for vdn_conn in vdn_conns]
            actions = np.array([content["actions"] for content in contents])
        else:
            obs = torch.from_numpy(obs).float()
            avail_action = torch.tensor(avail_action, dtype=torch.bool)
            with torch.no_grad():
                 q_values = utility_network(x=obs,avail_action =avail_action)
            actions  = torch.argmax(q_values,dim=-1)
        # Execute the action
        for i,vdn_conn in enumerate(vdn_conns):
                vdn_conn.send(("step",actions[i]))

        contents = [vdn_conn.recv() for vdn_conn in vdn_conns]
        next_obs = np.stack([content["next_obs"] for content in contents], axis=0)
        reward = np.array([content["reward"]   for content in contents])
        done = np.array([content["done"]     for content in contents])
        truncated = np.array([content["truncated"]     for content in contents])
        next_avail_action = np.stack([content["avail_actions"] for content in contents], axis=0)
        infos = [content.get("infos") for content in contents]
        
        step += args.num_envs

        ep_reward = [ep_reward[i] + reward[i] for i in range(args.num_envs) ] 
        ep_length = [ep + 1 for ep in ep_length]
        
        rb.store(obs, actions,reward,done,next_obs,next_avail_action) 
        
        obs = next_obs 
        avail_action = next_avail_action
        for i in range(args.num_envs):
            if done[i] or truncated[i]:
                vdn_conns[i].send(("reset",None))
                content = vdn_conns[i].recv()
                obs[i] = content["obs"]
                avail_action[i] = content["avail_actions"]
                ep_rewards.append(ep_reward[i])
                ep_lengths.append(ep_length[i])
                if args.env_type == 'smaclite':
                    ep_stats.append(infos[i])
                ep_reward[i] = 0
                ep_length[i] = 0
       

        if len(ep_rewards) > args.log_every :
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length",np.mean(ep_lengths),step)
            writer.add_scalar("rollout/epsilon",epsilon,step)
            if args.env_type == 'smaclite':
                writer.add_scalar("rollout/battle_won",np.mean([info["battle_won"] for info in ep_stats]), step)
            ep_rewards = []
            ep_lengths = []
            ep_stats   = []
               
        if step > args.learning_starts:
            if step % (args.train_freq* args.num_envs) ==0 : 
                batch_obs,batch_action,batch_reward,batch_next_obs,batch_next_avail_action,batch_done = rb.sample(args.batch_size)
                with torch.no_grad():
                    q_next_max,_ = target_network(batch_next_obs,avail_action =batch_next_avail_action ).max(dim=-1)
                vdn_q_max = q_next_max.sum(dim=-1)
                targets = batch_reward + args.gamma * (1-batch_done)*vdn_q_max 
                q_values = torch.gather(utility_network(batch_obs),dim=-1, index=batch_action.unsqueeze(-1)).squeeze()
                vqn_q_values = q_values.sum(dim = -1)
                loss = F.mse_loss(targets,vqn_q_values)
                optimizer.zero_grad()
                loss.backward()
                grads = [p.grad for p in utility_network.parameters() ]
                vdn_gradients = norm_d(grads,2)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(utility_network.parameters(), max_norm=args.clip_gradients)
                optimizer.step()
                num_updates+= 1
                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/grads", vdn_gradients, step)
                writer.add_scalar("train/num_updates", num_updates, step)
                
            if step % (args.target_network_update_freq*args.num_envs) == 0:
                        soft_update(
                            target_net=target_network,
                            utility_net=utility_network,
                            polyak=args.polyak
                        )
        if  step > 0 and  step % (args.eval_steps*args.num_envs) == 0 :
            eval_obs,_ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            while eval_ep < args.num_eval_ep:
                eval_obs = torch.from_numpy(eval_obs).float()
                mask_eval = torch.tensor(eval_env.get_avail_actions(), dtype=torch.bool)
                q_values = utility_network(eval_obs, avail_action = mask_eval)
                actions  = torch.argmax(q_values,dim=-1)
                next_obs_, reward, done, truncated, infos = eval_env.step(actions.cpu())
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
                




