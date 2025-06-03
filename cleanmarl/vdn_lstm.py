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
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
# * Compute the loss function
# TODO: seed, device, better title for the logger 


@dataclass
class Args:
    env_type: str = "smaclite"
    """ Pettingzoo, SMAClite ... """
    env_name: str = "MMM"
    """ Name of the environment"""
    env_family: str ="sisl"
    """ Env family when using pz"""
    buffer_size: int = 5000
    """ The size of the replay buffer"""
    seq_length: int = 10
    """ Length of the sequence to store in the buffer"""
    burn_in: int = 7
    """Sequences to burn during batch updates"""
    total_timesteps: int = 2000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    learning_starts: int = 5000 
    """ Number of env steps to initialize the replay buffer"""
    train_freq: int = 20
    """ Training frequency, relative to total_timesteps"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate: float =  0.00005
    """ Learning rate"""
    batch_size: int = 10
    """Batch size"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.025
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    hidden_dim: int = 64
    """ Hidden dimension"""
    num_layers: int = 2
    """ Number of layers"""
    target_network_update_freq: int = 10000
    """ Frequency of updating target network"""
    log_every: int = 1000
    """ Logging steps"""
    grad_clip: float =  4
    """grad clipping"""
    polyak: float = 1
    """polyak coefficient when using polyak averaging for target network update"""
    eval_steps: int = 20000
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
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRUCell(args.hidden_dim, hidden_dim)
        self.fc2 = nn.Sequential(nn.ReLU(),nn.Linear(hidden_dim, output_dim))        
    
    def forward(self,x,h=None,mask=None):
        x = self.fc1(x)
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        h = self.gru(x,h)
        x = self.fc2(h)
        if mask is not None:
            x = x.masked_fill(~mask, float('-inf'))
        return x,h
class ReplayBuffer:
    def __init__(self,buffer_size,num_agents,obs_space,action_space,seq_length):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.seq_length = seq_length

        self.obs = np.zeros((self.buffer_size,self.seq_length,self.num_agents,self.obs_space))
        self.mask = np.zeros((self.buffer_size,self.seq_length,self.num_agents,self.action_space))
        self.action = np.zeros((self.buffer_size,self.seq_length,self.num_agents))
        self.reward = np.zeros((self.buffer_size,self.seq_length))
        self.next_obs = np.zeros((self.buffer_size,self.seq_length,self.num_agents,self.obs_space))
        self.done = np.zeros((self.buffer_size,self.seq_length))
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
            torch.from_numpy(self.obs[indices]).float(),
            torch.from_numpy(self.action[indices]).long(),
            torch.from_numpy(rewards).float(),
            torch.from_numpy(self.next_obs[indices]).float(),
            torch.from_numpy(self.mask[indices]).bool(),
            torch.from_numpy(self.done[indices]).float()
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

if __name__ == "__main__":
    ## what if we periodically empty the replay buffer
    args = tyro.cli(Args)
    ## import the environment 
    kwargs = {} #{"render_mode":'human',"shared_reward":False}
    env = environment(env_type= args.env_type,
                      env_name=args.env_name,
                      env_family=args.env_family,
                      kwargs=kwargs)
    eval_env = environment(env_type= args.env_type,
                      env_name=args.env_name,
                      env_family=args.env_family,
                      kwargs=kwargs)
    ## initialize the utility and target networks
    agents_utility_network = Qnetwrok(input_dim=env.get_obs_size(),
                                          hidden_dim=args.hidden_dim,
                                          num_layer=args.num_layers,
                                          output_dim=env.get_action_size())
    agents_target_network = Qnetwrok(input_dim=env.get_obs_size(),
                                          hidden_dim=args.hidden_dim,
                                          num_layer=args.num_layers,
                                          output_dim=env.get_action_size())
    
    optimizer = getattr(optim, args.optimizer) # get which optimizer to use from args
    optimizer = optimizer(agents_utility_network.parameters(),lr = args.learning_rate)

    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space= env.get_obs_size(),
        action_space=env.get_action_size(),
        num_agents= env.n_agents,
        seq_length=args.seq_length
    )
    obs,_ = env.reset()
    h = None
    seq_to_rb = []
    seq_obs, seq_actions,seq_reward,seq_done,seq_next_obs,seq_mask = [],[],[],[],[],[]
    current_seq_len = 0
    mask = torch.tensor(env.get_avail_actions(), dtype=torch.bool, device=args.device)
    ep_reward = 0
    ep_length = 0
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
    for step in range(args.total_timesteps):
        ## select actions
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, step)
        if random.random() < epsilon:
            actions = env.sample()
            h = None
        else:
            obs = torch.from_numpy(obs).to(args.device)
            q_values,h = agents_utility_network(obs,h=h,mask =mask)
            h = h.detach()
            actions  = torch.argmax(q_values,dim=-1)
        next_obs, reward, done, truncated, infos = env.step(actions)
        ep_reward += reward
        ep_length += 1
        
        mask = torch.tensor(env.get_avail_actions(), dtype=torch.bool, device=args.device)

        if current_seq_len >= args.seq_length:
            current_seq_len = 0
            rb.store(np.stack(seq_obs),np.stack( seq_actions),np.stack(seq_reward),np.stack(seq_done),np.stack(seq_next_obs),np.stack(seq_mask) )
            seq_obs, seq_actions,seq_reward,seq_done,seq_next_obs,seq_mask = [],[],[],[],[],[]
        seq_obs.append(obs)
        seq_actions.append(actions)
        seq_reward.append(reward)
        seq_done.append(done)
        seq_next_obs.append(next_obs)
        seq_mask.append(mask)
        obs = next_obs 
        current_seq_len+= 1
        
        if done or truncated:
            obs, _ = env.reset()
            mask = torch.tensor(env.get_avail_actions(), dtype=torch.bool, device=args.device)
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            if args.env_type == 'smaclite':
                ep_stats.append(infos)
            ep_reward = 0
            ep_length = 0
            h = None

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
                h_target = None
                h_utility = None
                for t in range(args.burn_in):
                    batch_next_obs_t = batch_next_obs[:,t,:].reshape(args.batch_size*env.n_agents,-1)
                    mask_t = batch_mask[:,t,:].reshape(args.batch_size*env.n_agents,-1)
                    batch_obs_t = batch_obs[:,t,:].reshape(args.batch_size*env.n_agents,-1)
                    _,h_target =  agents_target_network(batch_next_obs_t,h = h_target,mask =mask_t )
                    _,h_utility = agents_utility_network(batch_obs_t,h=h_utility)
                h_target = h_target.detach()
                loss = 0
                for t in range(args.burn_in, args.seq_length):
                    with torch.no_grad():
                        batch_next_obs_t = batch_next_obs[:,t,:].reshape(args.batch_size*env.n_agents,-1)
                        mask_t = batch_mask[:,t,:].reshape(args.batch_size*env.n_agents,-1)
                        q_next,h_target = agents_target_network(batch_next_obs_t,h=h_target,mask =mask_t )
                        q_next = q_next.reshape(args.batch_size,env.n_agents,-1)
                        q_next_max,_ = q_next.max(dim=-1)
                        vdn_q_max = q_next_max.sum(dim=-1)
                    
                    targets = batch_reward[:,t].squeeze() + args.gamma * (1-batch_done[:,t].squeeze())*vdn_q_max
                    batch_obs_t = batch_obs[:,t,:].reshape(args.batch_size*env.n_agents,-1)
                    q_values,h_utility = agents_utility_network(batch_obs_t,h=h_utility)
                    q_values = q_values.reshape(args.batch_size,env.n_agents,-1)
                    q_values = torch.gather(q_values,dim=-1, index=batch_action[:,t,:].unsqueeze(-1)).squeeze()
                    vqn_q_values = q_values.sum(dim = -1)
                    loss += F.mse_loss(targets,vqn_q_values)
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
            eval_obs,_ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            h_eval = None
            while eval_ep < args.num_eval_ep:
                eval_obs = torch.from_numpy(eval_obs).to(args.device)
                mask_eval = torch.tensor(eval_env.get_avail_actions(), dtype=torch.bool, device=args.device)
                q_values,h_eval = agents_utility_network(eval_obs,h=h_eval, mask = mask_eval)
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
                    h_eval = None
            writer.add_scalar("eval/ep_reward",np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/std_ep_reward",np.std(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length",np.mean(eval_ep_length), step)
            if args.env_type == 'smaclite':
                writer.add_scalar("eval/battle_won",np.mean(np.mean([info["battle_won"] for info in eval_ep_stats])), step)
                

        

## Fix the sampling from replaybuffer





