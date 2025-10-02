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
    env_family: str ="mpe"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    gamma: float = 0.99
    """ Discount factor"""
    buffer_size: int = 5000
    """ The number of episodes in the replay buffer"""
    batch_size: int = 10
    """ Batch size"""
    normalize_reward: bool = True
    """ Normalize the rewards if True"""
    actor_hidden_dim: int = 32
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 64
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    train_freq: int = 1
    """ Train the network each «train_freq» step in the environment"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float =  0.0003
    """ Learning rate for the actor"""
    learning_rate_critic: float =  0.0003
    """ Learning rate for the critic"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    target_network_update_freq: int = 1
    """ Update the target network each target_network_update_freq» step in the environment"""
    polyak: float = 0.005
    """ Polyak coefficient when using polyak averaging for target network update"""
    log_every: int = 10
    """ Logging steps """
    eval_steps: int = 50
    """ Evaluate the policy each «eval_steps» steps"""
    num_eval_ep: int = 5
    """ Number of evaluation episodes"""
    device: str ="cpu"
    """ Device (cpu, gpu, mps)"""
    seed: int  = 42
    """ Random seed"""
    clip_gradients: int = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    tbptt:int = 10
    """Chunck size for Truncated Backpropagation Through Time tbptt"""



class Actor(nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Sequential(nn.ReLU(),nn.Linear(hidden_dim, output_dim))   
        

    
    def act(self,x,h,avail_action=None,hard=False):
        x,h = self.logits(x,h,avail_action)
        actions = F.gumbel_softmax(logits=x,hard=hard)
        return actions,h
    def logits(self,x,h,avail_action=None):
        x = self.fc1(x)
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        h = self.gru(x,h)
        x = self.fc2(h)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, float('-inf'))
        return x,h
    
class Critic(nn.Module):
    def __init__(self, input_dim,hidden_dim,num_layer,output_dim,num_agents) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, 1)))
    
    def forward(self,state,actions,grad_processing=False,batch_action=None):
        
        x = self.maddpg_inputs(state,actions,grad_processing,batch_action)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()
    def maddpg_inputs(self,state ,actions,grad_processing,batch_action):
        maddpg_inputs = torch.zeros((state.size(0),self.num_agents,self.input_dim))
        maddpg_inputs[:,:,:state.size(-1)] = state.unsqueeze(1)
        oh= actions.unsqueeze(1)
        oh = oh.expand(-1,self.num_agents,-1,-1)
        oh = oh.reshape(state.size(0),self.num_agents,-1)
        if grad_processing:
            b_oh= batch_action.unsqueeze(1)
            b_oh = b_oh.expand(-1,self.num_agents,-1,-1)
            b_oh = b_oh.reshape(state.size(0),self.num_agents,-1)
            mask = torch.eye(self.num_agents)
            mask = mask.unsqueeze(-1).expand(-1,-1,actions.size(-1))
            mask = mask.reshape(self.num_agents,-1)
            oh = torch.where(mask.bool(),oh, b_oh)
        maddpg_inputs[:,:,state.size(-1):] = oh
        return maddpg_inputs

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
        self.episodes[self.pos] = episode #{"obs": [],"actions":[],"reward":[],"states":[],"done":[],"avail_actions":[]}
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    def sample(self,batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = [self.episodes[i] for i in indices]
        lengths = [len(episode["obs"]) for episode in batch ]
        # print(lengths)
        max_length = max(lengths)
        obs = np.zeros((batch_size,max_length,self.num_agents,self.obs_space))
        avail_actions = np.zeros((batch_size,max_length,self.num_agents,self.action_space))
        actions = np.zeros((batch_size,max_length,self.num_agents,self.action_space))
        reward = np.zeros((batch_size,max_length))
        states = np.zeros((batch_size,max_length,self.state_space))
        done = np.ones((batch_size,max_length))
        mask = torch.zeros(batch_size, max_length,dtype=torch.bool)

        for i in range(batch_size):
            length = lengths[i]
            obs[i,:length] =np.stack(batch[i]["obs"])
            avail_actions[i,:length] =np.stack(batch[i]["avail_actions"])
            actions[i,:length] =np.stack(batch[i]["actions"])
            reward[i,:length] =np.stack(batch[i]["reward"])
            states[i,:length] =np.stack(batch[i]["states"])
            done[i,:length] =np.stack(batch[i]["done"])
            mask[i,:length] = 1

        if self.normalize_reward:
            mu = np.mean(reward[mask] )
            std = np.std(reward[mask] )
            reward[mask.bool()] = (reward[mask] - mu) /(std + 1e-6)
        
        return (
            torch.from_numpy(obs).float(),
            torch.from_numpy(actions).float(),
            torch.from_numpy(reward).float(),
            torch.from_numpy(states).float(),
            torch.from_numpy(avail_actions).bool(),
            torch.from_numpy(done).float(),
            mask,
        )




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
    ## what if we periodically empty the replay buffer
    args = tyro.cli(Args)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ## import the environment 
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
    
    actor = Actor(
        input_dim=env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        output_dim=env.get_action_size()
    )
    target_actor = Actor(
        input_dim=env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        output_dim=env.get_action_size()
    )

    maddpg_input_dim = env.get_state_size() +  env.n_agents*env.get_action_size()
    critic = Critic(
        input_dim=maddpg_input_dim,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        output_dim=env.get_action_size(),
        num_agents =  env.n_agents
    )
    target_critic = Critic(
        input_dim=maddpg_input_dim,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        output_dim=env.get_action_size(),
        num_agents =  env.n_agents
    )
    soft_update(
            target_net=target_critic,
            utility_net=critic,
            polyak=1.0
        )
    soft_update(
            target_net=target_actor,
            utility_net=actor,
            polyak=1.0
        )
    Optimizer = getattr(optim, args.optimizer) 
    actor_optimizer = Optimizer(actor.parameters(),lr = args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(),lr = args.learning_rate_critic)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    writer = SummaryWriter(f"runs/MADDPG-LSTM-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_space= env.get_obs_size(),
        state_space=env.get_state_size(),
        action_space=env.get_action_size(),
        num_agents= env.n_agents,
        normalize_reward= args.normalize_reward
    )
    num_episode = 0
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    step = 0
    while step < args.total_timesteps:
        episode = {"obs": [],"actions":[],"reward":[],"states":[],"done":[],"avail_actions":[]}
        obs, _ = env.reset()
        ep_reward, ep_length = 0,0
        done, truncated = False, False
        h = None
        while not done and not truncated:
            obs = torch.from_numpy(obs).to(args.device).float()
            avail_action = torch.tensor(env.get_avail_actions(), dtype=torch.bool, device=args.device)
            state = torch.from_numpy(env.get_state()).to(args.device).float()
            with torch.no_grad():
                actions,h = actor.act(obs,h,avail_action =avail_action,hard=True) ## These are one hot-vectors
                actions_to_take = torch.argmax(actions,dim=-1)
            
            next_obs, reward, done, truncated, infos = env.step(actions_to_take)
            
            ep_reward += reward
            ep_length += 1
            step +=1
            episode["obs"].append(obs)
            episode["actions"].append(actions)
            episode["reward"].append(reward)
            episode["done"].append(done)
            episode["avail_actions"].append(avail_action)
            episode["states"].append(state)
            # print(done)
            obs = next_obs 
        # print(episode["done"])
        rb.store(episode)
        # print("ep_length",ep_length)
        num_episode += 1
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)
        if args.env_type == 'smaclite':
            ep_stats.append(infos) ## Add battle won for smaclite

        if num_episode % args.log_every == 0:
                if len(ep_rewards) > 0: 
                    writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
                    writer.add_scalar("rollout/ep_length",np.mean(ep_lengths),step)
                    if args.env_type == 'smaclite':
                        writer.add_scalar("rollout/battle_won",np.mean(np.mean([info["battle_won"] for info in ep_stats])), step)
                    ep_rewards = []
                    ep_lengths = []
                    ep_stats   = []
        # print("num_episode",num_episode)
        if num_episode > args.batch_size:
            # print("I'm in ",num_episode)
            if num_episode % args.train_freq == 0:
                batch_obs,batch_action,batch_reward,batch_states,batch_avail_action,batch_done, batch_mask = rb.sample(args.batch_size)
                ## train the critic
                critic_loss = 0
                h_targ = None
                # print(batch_obs.shape)
                for t in range(batch_obs.size(1)-1):
                    with torch.no_grad():
                        b_obs_t1 = batch_obs[:,t+1].reshape(args.batch_size*eval_env.n_agents,-1)
                        b_avail_actions_t1 = batch_avail_action[:,t+1].reshape(args.batch_size*eval_env.n_agents,-1)
                        actions_from_target_actor,h_targ = target_actor.act(b_obs_t1,h_targ,avail_action =b_avail_actions_t1,hard=True)
                        actions_from_target_actor = actions_from_target_actor.reshape(args.batch_size,eval_env.n_agents,-1)
                        qvals_from_taget_critic = target_critic(batch_states[:,t+1],actions_from_target_actor)
                        qvals_from_taget_critic = torch.nan_to_num(qvals_from_taget_critic, nan=0.0)
                    targets = batch_reward[:,t].unsqueeze(-1).expand(-1,env.n_agents) + args.gamma * (1-batch_done[:,t+1].unsqueeze(-1).expand(-1,env.n_agents))*qvals_from_taget_critic
                    q_values =critic(batch_states[:,t],batch_action[:,t])
                    critic_loss += F.mse_loss(targets[batch_mask[:,t]],q_values[batch_mask[:,t]]) * batch_mask[:,t].sum()
                critic_loss /= batch_mask.sum()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_gradients = norm_d([p.grad for p in critic.parameters() ],2)
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.clip_gradients)
                critic_optimizer.step()
                
                if num_episode % args.target_network_update_freq == 0:
                    soft_update(
                        target_net=target_critic,
                        utility_net=critic,
                        polyak=args.polyak)
                ## train the actor
                actor_losses = 0
                actor_gradients = 0
                h_actor = None
                truncated_actor_loss = None
                actor_loss_denominator = None
                for t in range(batch_obs.size(1)):
                    b_obs_t = batch_obs[:,t].reshape(args.batch_size*eval_env.n_agents,-1)
                    b_avail_actions_t = batch_avail_action[:,t].reshape(args.batch_size*eval_env.n_agents,-1)
                    actions,h_actor =  actor.act(b_obs_t,h_actor,avail_action =b_avail_actions_t,hard=True)
                    actions = actions.reshape(args.batch_size,eval_env.n_agents,-1)
                    qvals = critic(batch_states[:,t],actions,grad_processing=True,batch_action=batch_action[:,t])
                    actor_loss=  -qvals[batch_mask[:,t]].sum()
                    actor_losses += actor_loss
                    if truncated_actor_loss is  None:
                            truncated_actor_loss = actor_loss
                            actor_loss_denominator = batch_mask[:,t].sum()
                    else:
                        truncated_actor_loss += actor_loss
                        actor_loss_denominator += batch_mask[:,t].sum()
                    if ((t+1) % args.tbptt == 0) or (t == (batch_obs.size(1)-1)):
                        truncated_actor_loss = truncated_actor_loss/actor_loss_denominator
                        actor_optimizer.zero_grad()
                        truncated_actor_loss.backward()
                        tbptt_actor_gradients = norm_d([p.grad for p in actor.parameters() ],2)
                        actor_gradients += tbptt_actor_gradients
                        if args.clip_gradients > 0:
                            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.clip_gradients)
                        actor_optimizer.step()
                        truncated_actor_loss = None
                        h_actor = h_actor.detach()


                if num_episode % args.target_network_update_freq == 0:
                    soft_update(
                        target_net=target_actor,
                        utility_net=actor,
                        polyak=args.polyak)
                    
                writer.add_scalar("train/critic_loss", critic_loss.item(), step)
                writer.add_scalar("train/critic_gradients", critic_gradients, step)
                writer.add_scalar("train/actor_loss", actor_losses/batch_mask.sum(), step)
                writer.add_scalar("train/actor_gradients", actor_gradients/batch_obs.size(1), step)
                
        
        
        

        if num_episode % args.eval_steps == 0:
            eval_obs,_ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            h_eval = None 
            while eval_ep < args.num_eval_ep:
                eval_obs = torch.from_numpy(eval_obs).to(args.device).float()
                mask_eval = torch.tensor(eval_env.get_avail_actions(), dtype=torch.bool, device=args.device)
                with torch.no_grad():
                    logits,h_eval = actor.logits(eval_obs,h_eval,avail_action = mask_eval)
                    eval_actions  = torch.argmax(logits,dim=-1)
                next_obs_, reward, done, truncated, infos = eval_env.step(eval_actions)
                current_reward += reward
                current_ep_length += 1
                eval_obs = next_obs_
                if done or truncated:
                    eval_obs, _ = eval_env.reset()
                    h_eval = None
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
                
