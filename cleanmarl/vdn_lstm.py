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
    env_type: str = "smaclite"
    """ Pettingzoo, SMAClite ... """
    env_name: str = "3m"
    """ Name of the environment"""
    env_family: str ="mpe"
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
    train_freq: int = 10
    """ Train the network each «train_freq» step in the environment. The used value is train_freq*num_envs"""
    optimizer: str = "AdamW"
    """ The optimizer"""
    learning_rate: float =  0.0008
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
    device: str ="cpu"
    """ Device (cpu, cuda, mps)"""
    seed: int  = 1
    """ Random seed"""

    

class Qnetwrok(nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Sequential(nn.ReLU(),nn.Linear(hidden_dim, output_dim))        
    
    def forward(self,x,h=None,avail_action=None):
        x = self.fc1(x)
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        h = self.gru(x,h)
        x = self.fc2(h)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, float('-inf'))
        return x,h
class ReplayBuffer:
    def __init__(self,buffer_size,num_agents,obs_space,action_space,seq_length,normalize_reward= False,device="cpu"):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.seq_length = seq_length
        self.normalize_reward = normalize_reward
        self.device = device

        self.obs = np.zeros((self.buffer_size,self.seq_length,self.num_agents,self.obs_space))
        self.action = np.zeros((self.buffer_size,self.seq_length,self.num_agents))
        self.reward = np.zeros((self.buffer_size,self.seq_length))
        self.next_obs = np.zeros((self.buffer_size,self.seq_length,self.num_agents,self.obs_space))
        self.next_avail_action = np.zeros((self.buffer_size,self.seq_length,self.num_agents,self.action_space))
        self.done = np.zeros((self.buffer_size,self.seq_length))
        self.pos = 0
        self.size = 0
        self.last_pos = 0
    def store(self,obs,action,reward,done,next_obs,next_avail_action,is_last=False):
        if is_last:
            toadd = self.seq_length  - len(obs)
            obs = np.concatenate((self.obs[self.last_pos][-toadd:],obs),axis=0)
            action = np.concatenate((self.action[self.last_pos][-toadd:],action),axis=0)
            reward = np.concatenate((self.reward[self.last_pos][-toadd:],reward),axis=0)
            done = np.concatenate((self.done[self.last_pos][-toadd:],done),axis=0)
            next_obs = np.concatenate((self.next_obs[self.last_pos][-toadd:],next_obs),axis=0)
            next_avail_action = np.concatenate((self.next_avail_action[self.last_pos][-toadd:],next_avail_action),axis=0)
            
        self.obs[self.pos] = obs
        self.action[self.pos] = action
        self.reward[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.next_avail_action[self.pos]  = next_avail_action
        self.done[self.pos] = done
        self.last_pos = self.pos
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
            torch.from_numpy(self.obs[indices]).float().to(self.device),
            torch.from_numpy(self.action[indices]).long().to(self.device),
            torch.from_numpy(rewards).float().to(self.device),
            torch.from_numpy(self.next_obs[indices]).float().to(self.device),
            torch.from_numpy(self.next_avail_action[indices]).bool().to(self.device),
            torch.from_numpy(self.done[indices]).float().to(self.device)
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

if __name__ == "__main__":
    
    args = tyro.cli(Args)
    # Set the randomness seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    kwargs = {} #{"render_mode":'human',"shared_reward":False}
    env = environment(env_type= args.env_type,
                      env_name=args.env_name,
                      env_family=args.env_family,
                       agent_ids=args.agent_ids,
                      kwargs=kwargs)
    eval_env = environment(env_type= args.env_type,
                      env_name=args.env_name,
                      env_family=args.env_family,
                       agent_ids=args.agent_ids,
                      kwargs=kwargs)
    ## initialize the utility and target networks
    utility_network = Qnetwrok(input_dim=env.get_obs_size(),
                                          hidden_dim=args.hidden_dim,
                                          output_dim=env.get_action_size()).to(device)
    target_network = copy.deepcopy(utility_network).to(device)
    
    optimizer = getattr(optim, args.optimizer) # get which optimizer to use from args
    optimizer = optimizer(utility_network.parameters(),lr = args.learning_rate)

    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space= env.get_obs_size(),
        action_space=env.get_action_size(),
        num_agents= env.n_agents,
        seq_length=args.seq_length,
        normalize_reward= args.normalize_reward,
        device = device
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
                name=f'VDN-{run_name}'
            )
    writer = SummaryWriter(f"runs/VDN-lstm-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    obs,_ = env.reset()
    avail_action = env.get_avail_actions()
    h = None
    seq_obs, seq_actions,seq_reward,seq_done,seq_next_obs,seq_next_avail_action = [],[],[],[],[],[]
    current_seq_len = 0
    ep_reward = 0
    ep_length = 0
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    num_updates=0
    num_episodes = 0
    for step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, step)
        # We always need the forward pass even when taking random actions in order to let the h flow through time
        with torch.no_grad():
            q_values,h = utility_network(x=torch.from_numpy(obs).float().to(args.device),
                                         h=h,
                                         avail_action =torch.tensor(avail_action, dtype=torch.bool).to(device))
        if random.random() < epsilon:
            actions = env.sample()
        else:
            actions  = torch.argmax(q_values,dim=-1).cpu()
        next_obs, reward, done, truncated, infos = env.step(actions)
        next_avail_action = env.get_avail_actions() # We need the next_avail_action to compute the target loss : max of Q(next_state)
        
        ep_reward += reward
        ep_length += 1

        seq_obs.append(obs)
        seq_actions.append(actions)
        seq_reward.append(reward)
        seq_done.append(done)
        seq_next_obs.append(next_obs)
        seq_next_avail_action.append(next_avail_action)
        current_seq_len+= 1

        obs = next_obs 
        avail_action = next_avail_action 

        if current_seq_len == args.seq_length:
            rb.store(np.stack(seq_obs),np.stack( seq_actions),np.stack(seq_reward),np.stack(seq_done),np.stack(seq_next_obs),np.stack(seq_next_avail_action) )
            current_seq_len = 0
            seq_obs, seq_actions,seq_reward,seq_done,seq_next_obs,seq_next_avail_action = [],[],[],[],[],[]
        
        if done or truncated:
            obs, _ = env.reset()
            avail_action = env.get_avail_actions()
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            if args.env_type == 'smaclite':
                ep_stats.append(infos)
            ep_reward = 0
            ep_length = 0
            num_episodes+=1
            h = None
            if 0 < current_seq_len and  current_seq_len< args.seq_length:
                current_seq_len = 0
                rb.store(np.stack(seq_obs),np.stack( seq_actions),np.stack(seq_reward),np.stack(seq_done),np.stack(seq_next_obs),np.stack(seq_next_avail_action),is_last=True )
                seq_obs, seq_actions,seq_reward,seq_done,seq_next_obs,seq_next_avail_action = [],[],[],[],[],[]
        


        if step > args.learning_starts:
            if step % args.train_freq == 0: 
                batch_obs,batch_action,batch_reward,batch_next_obs,batch_next_avail_action,batch_done = rb.sample(args.batch_size)
                h_target = None
                h_utility = None
                with torch.no_grad():
                    for t in range(args.burn_in):
                        batch_next_obs_t = batch_next_obs[:,t,:].reshape(args.batch_size*env.n_agents,-1)
                        batch_obs_t = batch_obs[:,t,:].reshape(args.batch_size*env.n_agents,-1)
                        _,h_target =  target_network(batch_next_obs_t,h = h_target)
                        _,h_utility = utility_network(batch_obs_t,h=h_utility)
                loss = 0
                for t in range(args.burn_in, args.seq_length):
                    with torch.no_grad():
                        batch_next_obs_t = batch_next_obs[:,t,:].reshape(args.batch_size*env.n_agents,-1)
                        batch_next_avail_action_t = batch_next_avail_action[:,t,:].reshape(args.batch_size*env.n_agents,-1)
                        q_next,h_target = target_network(batch_next_obs_t,h=h_target,avail_action =batch_next_avail_action_t )
                        q_next = q_next.reshape(args.batch_size,env.n_agents,-1)
                        q_next_max,_ = q_next.max(dim=-1)
                        vdn_q_max = q_next_max.sum(dim=-1)
                        targets = batch_reward[:,t] + args.gamma * (1-batch_done[:,t])*vdn_q_max

                    batch_obs_t = batch_obs[:,t,:].reshape(args.batch_size*env.n_agents,-1)
                    q_values,h_utility = utility_network(batch_obs_t,h=h_utility)
                    q_values = q_values.reshape(args.batch_size,env.n_agents,-1)
                    q_values = torch.gather(q_values,dim=-1, index=batch_action[:,t,:].unsqueeze(-1)).squeeze()
                    vqn_q_values = q_values.sum(dim = -1)
                    loss += F.mse_loss(targets,vqn_q_values)
                loss = loss / (args.seq_length - args.burn_in)
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

            if step % args.target_network_update_freq == 0:
                    soft_update(
                        target_net=target_network,
                        utility_net=utility_network,
                        polyak=args.polyak
                    )
        
        if len(ep_rewards) > args.log_every :
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length",np.mean(ep_lengths),step)
            writer.add_scalar("rollout/epsilon",epsilon,step)
            writer.add_scalar("rollout/num_episodes",num_episodes,step)
            if args.env_type == 'smaclite':
                writer.add_scalar("rollout/battle_won",np.mean([info["battle_won"] for info in ep_stats]), step)
            ep_rewards = []
            ep_lengths = []
            ep_stats   = []
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
                q_values,h_eval = utility_network(torch.from_numpy(eval_obs).float().to(device),
                                                  h=h_eval, 
                                                  avail_action = torch.tensor(eval_env.get_avail_actions(), dtype=torch.bool, device=device))
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
                writer.add_scalar("eval/battle_won",np.mean([info["battle_won"] for info in eval_ep_stats]), step)
                

    writer.close()
    if args.use_wnb:
        wandb.finish()
    env.close()
    eval_env.close()


