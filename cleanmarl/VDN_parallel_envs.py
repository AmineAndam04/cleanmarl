from multiprocessing import Pipe, Process
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
import tyro
import random 
from env.pettingzoo_wrapper import PettingZooWrapper
from env.smaclite_wrapper import SMACliteWrapper
import torch.nn.functional as F
import datetime
from torch.utils.tensorboard import SummaryWriter



@dataclass
class Args:
    env_type: str = "smaclite"
    """ Pettingzoo, SMAClite ... """
    env_name: str = "MMM"
    """ Name of the environment"""
    env_family: str ="mpe"
    """ Env family when using pz"""
    num_envs: int  = 8
    buffer_size: int = 5000
    """ The size of the replay buffer"""
    total_timesteps: int = 2000000
    gamma: float = 0.99
    """ Discount factor"""
    """ Total steps in the environment during training"""
    learning_starts: int = 5000 
    """ Number of env steps to initialize the replay buffer"""
    train_freq: int = 64 # choose train_freq % num_envs == 0
    """ Training frequency, relative to total_timesteps"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate: float =  0.00001
    """ Learning rate"""
    batch_size: int = 8
    """Batch size"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.05
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    hidden_dim: int = 128
    """ Hidden dimension"""
    num_layers: int = 2
    """ Number of layers"""
    target_network_update_freq: int = 5000
    """ Frequency of updating target network"""
    log_every: int = 3000
    """ Logging steps"""
    grad_clip: float =  4
    """grad clipping"""
    polyak: float = 1
    """polyak coefficient when using polyak averaging for target network update"""
    eval_steps: int = 8000
    """ Evaluate the policy each eval_steps steps"""
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""
    device: str ="cpu"
    """ Device (cpu, gpu, mps)"""
    normalize_reward: bool = True
    """ Normalize the rewards"""

    

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
        

    
    def forward(self,x,mask=None):
        for layer in self.layers:
            x = layer(x)
        if mask is not None:
            x = x.masked_fill(~mask, float('-inf'))
        return x
class ReplayBuffer:
    def __init__(self,buffer_size,num_agents,obs_space,action_space,num_envs):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_envs = num_envs

        self.obs = np.zeros((self.buffer_size,self.num_envs,self.num_agents,self.obs_space))
        self.mask = np.zeros((self.buffer_size,self.num_envs,self.num_agents,self.action_space))
        self.action = np.zeros((self.buffer_size,self.num_envs,self.num_agents))
        self.reward = np.zeros((self.buffer_size,self.num_envs))
        self.next_obs = np.zeros((self.buffer_size,self.num_envs,self.num_agents,self.obs_space))
        self.done = np.zeros((self.buffer_size,self.num_envs))
        self.pos = 0
        self.size = 0
    def store(self,obs,action,reward,done,next_obs,mask):
        self.obs[self.pos] = obs
        self.action[self.pos] = action
        self.reward[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.mask[self.pos]  = mask
        self.done[self.pos] = done
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    def sample(self,batch_size,normalize_reward= False):
        indices = np.random.randint(0, self.size, size=batch_size)
        if normalize_reward:
            mu = np.mean(self.reward)
            std = np.std(self.reward)
            rewards = (self.reward[indices] - mu) /(std + 1e-6)
        else :
            rewards  = self.reward[indices]
        return (
            torch.from_numpy(self.obs[indices]).reshape(batch_size*self.num_envs,self.num_agents,self.obs_space).float(),
            torch.from_numpy(self.action[indices]).reshape(batch_size*self.num_envs,self.num_agents).long(),
            torch.from_numpy(rewards).reshape(batch_size*self.num_envs,1).float(),
            torch.from_numpy(self.next_obs[indices]).reshape(batch_size*self.num_envs,self.num_agents,self.obs_space).float(),
            torch.from_numpy(self.mask[indices]).reshape(batch_size*self.num_envs,self.num_agents,self.action_space).bool(),
            torch.from_numpy(self.done[indices]).reshape(batch_size*self.num_envs,1).float()
        )

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def environment(env_type, env_name, env_family,kwargs):
    if env_type == 'pz':
        env = PettingZooWrapper(family = env_family, env_name = env_name,**kwargs)
    elif env_type == 'smaclite':
        env = SMACliteWrapper(map_name=env_name)
    
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
        
def closest_multiple(a, b):
    if a % b == 0:
        return a
    else:
        lower = a - (a % b)
        upper = lower + b
        # Choose the closer of the two
        return lower if abs(a - lower) <= abs(a - upper) else upper

if __name__ == "__main__":
    ## what if we periodically empty the replay buffer
    args = tyro.cli(Args)
    args.train_freq = closest_multiple(args.train_freq,args.num_envs)
    args.target_network_update_freq = closest_multiple(args.target_network_update_freq,args.num_envs)
    args.eval_steps = closest_multiple(args.eval_steps,args.num_envs)
    kwargs = {} #{"render_mode":'human',"shared_reward":False}
    ## Create the pipes to communicate between the main process (VDN algorithm) and child processes (envs)
    conns = [Pipe() for _ in range(args.num_envs)]
    vdn_conns, env_conns = zip(*conns)
    envs = [CloudpickleWrapper(environment(env_type= args.env_type,
                      env_name=args.env_name,
                      env_family=args.env_family,
                      kwargs=kwargs)) for _ in range(args.num_envs)]
    processes = [Process(
        target=env_worker,
        args=(env_conns[i],envs[i]))
        for i in range(args.num_envs)]
    for process in processes:
            process.daemon = True
            process.start()
    vdn_conns[0].send(("get_env_info", None))
    env_info = vdn_conns[0].recv()
    print(env_info)
    
    eval_env = environment(env_type= args.env_type,
                      env_name=args.env_name,
                      env_family=args.env_family,
                      kwargs=kwargs)
    
    agents_utility_network = Qnetwrok(input_dim=env_info["obs_size"],
                                          hidden_dim=args.hidden_dim,
                                          num_layer=args.num_layers,
                                          output_dim=env_info["action_size"])
    agents_target_network = Qnetwrok(input_dim=env_info["obs_size"],
                                          hidden_dim=args.hidden_dim,
                                          num_layer=args.num_layers,
                                          output_dim=env_info["action_size"])
    ## initialize the optimizer
    optimizer = getattr(optim, args.optimizer) # get which optimizer to use from args
    optimizer = optimizer(agents_utility_network.parameters(),lr = args.learning_rate)

    ## initialize a shared replay buffer
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space= env_info["obs_size"],
        action_space=env_info["action_size"],
        num_agents= env_info["n_agents"],
        num_envs = args.num_envs
    )
    obs = []
    avail_actions = []
    for vdn_conn in vdn_conns:
        vdn_conn.send(("reset",None))
        content = vdn_conn.recv()
        obs.append(content["obs"])
        avail_actions.append(content["avail_actions"])
    mask = torch.tensor(np.array(avail_actions), dtype=torch.bool, device=args.device)
    # Ò
    ep_reward = np.zeros(args.num_envs)
    ep_length = np.zeros(args.num_envs)
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    losses = []
    q_vals = []
    gradients = []
    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    step = 0
    while step < args.total_timesteps:
        
        ## select actions
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, step)
        if random.random() < epsilon:
            actions = []
            for vdn_conn in vdn_conns:
                vdn_conn.send(("sample",None))
                content = vdn_conn.recv()
                actions.append(content["actions"])
            actions = np.array(actions)
        else:
            ## instead of looping through the agents, we can see the number of the agents as a batch size (env.n_agents, shape_obs) ---> (batch_size, shape_obs)
            obs = torch.from_numpy(obs).to(args.device)
            obs = obs.view(args.num_envs * env_info["n_agents"], -1)
            mask = mask.view(args.num_envs * env_info["n_agents"], -1)
            q_values = agents_utility_network(obs,mask =mask)
            q_values = q_values.view(args.num_envs, env_info["n_agents"], -1)
            actions  = torch.argmax(q_values,dim=-1)
            obs = obs.view(args.num_envs,env_info["n_agents"],-1).cpu().numpy()
        next_obs, reward, done, truncated, infos,mask = [],[],[],[],[],[]
        for i,vdn_conn in enumerate(vdn_conns):
                vdn_conn.send(("step",actions[i]))
                content = vdn_conn.recv()
                next_obs.append(content["next_obs"])
                reward.append(content["reward"])
                done.append(content["done"])
                truncated.append(content["truncated"])
                infos.append(content["infos"])
                mask.append(content["avail_actions"])
        step += args.num_envs
        ep_reward = [ep_reward[i] + reward[i] for i in range(args.num_envs) ] 
        ep_length = [ep + 1 for ep in ep_length]
        
        rb.store(obs, actions,reward,done,np.array(next_obs),mask)
        
        obs = np.array(next_obs)
        
        for i in range(args.num_envs):
            if done[i] or truncated[i]:
                vdn_conns[i].send(("reset",None))
                content = vdn_conns[i].recv()
                obs[i] = content["obs"]
                mask[i] = content["avail_actions"]
                ep_rewards.append(ep_reward[i])
                ep_lengths.append(ep_length[i])
                if args.env_type == 'smaclite':
                    ep_stats.append(infos[i])
                ep_reward[i] = 0
                ep_length[i] = 0
        mask = torch.tensor(mask, dtype=torch.bool, device=args.device)

        if step % args.log_every == 0:
                if len(ep_rewards) > 0: 
                    writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
                    writer.add_scalar("rollout/ep_length",np.mean(ep_lengths),step)
                    if args.env_type == 'smaclite':
                        writer.add_scalar("rollout/battle_won",np.mean(np.mean([info["battle_won"] for info in ep_stats])), step)
                    ep_rewards = []
                    ep_lengths = []
                    ep_stats   = []
               
        if step > args.learning_starts:
            if step % args.train_freq: 
                batch_obs,batch_action,batch_reward,batch_next_obs,batch_mask,batch_done = rb.sample(args.batch_size,normalize_reward= args.normalize_reward)
                
                with torch.no_grad():
                    #print("batch_next_obs.shape",batch_next_obs.shape)
                    #print("batch_mask.shape",batch_mask.shape)
                    q_next_max,_ = agents_target_network(batch_next_obs,mask =batch_mask ).max(dim=-1)
                    #print("q_next_max.shape",q_next_max.shape)
                    vdn_q_max = q_next_max.sum(dim=-1)
                    #print("vdn_q_max.shape",vdn_q_max.shape)
                    #print("batch_reward.shape",batch_reward.shape)
                    #print("batch_done.shape",batch_done.shape)
                    targets = batch_reward.squeeze() + args.gamma * (1-batch_done.squeeze())*(torch.sum(q_next_max, dim=-1))
                    #print("targets.shape",targets.shape)
                #print("batch_obs.shape",batch_obs.shape)
                q_values = torch.gather(agents_utility_network(batch_obs),dim=-1, index=batch_action.unsqueeze(-1)).squeeze()
                vqn_q_values = q_values.sum(dim = -1)
                #print("vqn_q_values.shape",vqn_q_values.shape)
                loss = F.mse_loss(targets,vqn_q_values)
                optimizer.zero_grad()
                loss.backward()
                grads = [p.grad for p in agents_utility_network.parameters() ]
                grad_norm_2 = norm_d(grads,2)
                gradients.append(grad_norm_2)
                torch.nn.utils.clip_grad_norm_(agents_utility_network.parameters(), max_norm=args.grad_clip, norm_type=2)
                optimizer.step()
                losses.append(loss.item())
                q_vals.append(vqn_q_values.mean().item())
                if step % args.target_network_update_freq:
                    soft_update(
                        target_net=agents_target_network,
                        utility_net=agents_utility_network,
                        polyak=args.polyak
                    )
            if step % args.log_every == 0:
                writer.add_scalar("train/loss", np.mean(losses), step)
                writer.add_scalar("train/q_values", np.mean(q_vals), step)
                writer.add_scalar("train/grads", np.mean(gradients), step)
                losses = []
                q_vals = []
                gradients = []

        if step % args.eval_steps == 0 and step > args.learning_starts:
            eval_obs,_ = eval_env.reset(seed=random.randint(0, 100000))
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            while eval_ep < args.num_eval_ep:
                eval_obs = torch.from_numpy(eval_obs).to(args.device)
                mask_eval = torch.tensor(eval_env.get_avail_actions(), dtype=torch.bool, device=args.device)
                q_values = agents_utility_network(eval_obs, mask = mask_eval)
                actions  = torch.argmax(q_values,dim=-1)
                next_obs_, reward, done, truncated, infos = eval_env.step(actions)
                current_reward += reward
                current_ep_length += 1
                if done or truncated:
                    eval_obs, _ = eval_env.reset()
                    eval_ep_reward.append(current_reward)
                    eval_ep_length.append(current_ep_length)
                    eval_ep_stats.append(infos)
                    current_reward = 0
                    current_ep_length = 0
                    eval_ep +=1
                eval_obs = next_obs_
            writer.add_scalar("eval/ep_reward",np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/std_ep_reward",np.std(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length",np.mean(eval_ep_length), step)
            if args.env_type == 'smaclite':
                writer.add_scalar("eval/battle_won",np.mean(np.mean([info["battle_won"] for info in eval_ep_stats])), step)
                

        

## Fix the sampling from replaybuffer





