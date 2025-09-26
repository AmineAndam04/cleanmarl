# TODO : accumulate gradient
##What worked: unormalize reward and normalize advantage
import torch
import tyro
import datetime
import random 
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import torch.nn.functional as F
from multiprocessing import Pipe, Process
from env.pettingzoo_wrapper import PettingZooWrapper
from env.smaclite_wrapper import SMACliteWrapper
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
    critic_hidden_dim: int = 128
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 2
    """ Number of hidden layers of critic network"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float =  0.0005
    """ Learning rate"""
    learning_rate_critic: float =  0.0005
    """ Learning rate"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    td_lambda: float = 0.95
    """ TD() discount factor"""
    device: str ="cpu"
    """ Device (cpu, gpu, mps)"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    target_network_update_freq: int = 150
    """ Update the target network each target_network_update_freq» step in the environment"""
    polyak: float = 1
    """ Polyak coefficient when using polyak averaging for target network update"""
    eval_steps: int = 10
    """ Evaluate the policy each «eval_steps» steps"""
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""
    entropy_coef: float = 0.001
    """ Entropy coefficient """
    nsteps : int = 1

    start_e: float = 0.5
    """ The starting value of epsilon, for exploration"""
    end_e: float = 0.002
    """ The end value of epsilon, for exploration"""
    exploration_fraction: float = 750
    """ The fraction of «total-timesteps» it takes from to go from start_e to  end_e"""
    seed: int  = 1
    tbptt:int = 1

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
        self.episodes[self.pos] = episode #(obs,action,reward,done,next_obs,mask)
        self.pos += 1
    def get_batch(self):
        self.pos = 0
        lengths = [len(episode["obs"]) for episode in self.episodes ]
        max_length = max(lengths)
        obs = np.zeros((self.buffer_size,max_length,self.num_agents,self.obs_space))
        avail_actions = np.zeros((self.buffer_size,max_length,self.num_agents,self.action_space))
        actions = np.zeros((self.buffer_size,max_length,self.num_agents))
        reward = np.zeros((self.buffer_size,max_length))
        states = np.zeros((self.buffer_size,max_length,self.state_space))
        done = np.zeros((self.buffer_size,max_length))
        mask = torch.zeros(self.buffer_size, max_length,dtype=torch.bool)
        for i in range(self.buffer_size):
            length = lengths[i]
            obs[i,:length] = np.stack(self.episodes[i]["obs"])
            avail_actions[i,:length] = np.stack(self.episodes[i]["avail_actions"])
            actions[i,:length] = np.stack(self.episodes[i]["actions"])
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
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))
        

    
    def act(self,x,eps=0,avail_action=None):
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, float('-inf'))
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        # print("avail_action",avail_action)
        # print("shape",avail_action.shape)
        masked_eps = (avail_action)* (eps/avail_action.sum(dim=-1,keepdim=True))
        probs = (1-eps)*F.softmax(x,dim=-1) + masked_eps
        distribution = Categorical(probs)
        action = distribution.sample()
        return action
    def logits(self,x,eps=0,avail_action=None):
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, -1e9)
        # x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        masked_eps = (avail_action)* (eps/avail_action.sum(dim=-1,keepdim=True))
        probs = (1-eps)*F.softmax(x,dim=-1) + masked_eps
        return probs
    
class Critic(nn.Module):
    def __init__(self, input_dim,hidden_dim,num_layer,output_dim,num_agents) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))
        

    
    def forward(self,state ,observations,actions,avail_actions=None):
        if state.dim() < 2:
            state = state.unsqueeze(0)
            observations = observations.unsqueeze(0)
            actions = actions.unsqueeze(0)
            if avail_actions is not None:
                avail_actions = avail_actions.unsqueeze(0)

        x = self.coma_inputs(state ,observations,actions)
        for layer in self.layers:
            x = layer(x)
        if avail_actions is not None:
            x = x.masked_fill(~avail_actions, float('-inf'))
        return x.squeeze()
    def coma_inputs(self,state ,observations,actions):
        coma_inputs = torch.zeros((state.size(0),self.num_agents,self.input_dim))

        # print("coma_inputs",coma_inputs.shape)
        # print("state",state.shape)
        # print("observations",observations.shape)
        # print("actions",actions.shape)
        coma_inputs[:,:,:state.size(-1)] = state.unsqueeze(1)
        coma_inputs[:,:,state.size(-1):state.size(-1)+observations.size(-1)] = observations
        # one-hot encode actions
        one_hot = F.one_hot(actions.long(), num_classes=self.output_dim).float()  # (B, n_agents, A)
        # print("one_hot",one_hot.shape)
        # print("actions",actions.shape)
        # we need, for each agent i, a concatenation of other agents' one-hot actions
        mask = ~torch.eye(self.num_agents, dtype=torch.bool)  # (n_agents, n_agents)
        # expand one_hot to (B, n_agents, n_agents, A) then select masked entries
        oh = one_hot.unsqueeze(1).expand(state.size(0), self.num_agents, self.num_agents, self.output_dim)
        oh = oh[mask.unsqueeze(0).expand(state.size(0), -1, -1)]  # (B * n_agents * (n_agents-1), A)
        oh = oh.view(state.size(0), self.num_agents, (self.num_agents - 1) * self.output_dim)
        coma_inputs[:,:,state.size(-1)+observations.size(-1):] = oh
        # mask = ~torch.eye(self.num_agents, dtype=torch.bool)  
        # actions_ = actions.unsqueeze(1).expand(state.size(0), self.num_agents, self.num_agents)
        # actions_ = actions_[mask.unsqueeze(0).expand(state.size(0), -1, -1)]
        # actions_ = actions_.view(state.size(0), self.num_agents, self.num_agents - 1)
        
        # coma_inputs[:,:,state.size(-1)+observations.size(-1):] = actions_
        return coma_inputs




def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def environment(env_type, env_name, env_family,agent_ids,kwargs):
    if env_type == 'pz':
        env = PettingZooWrapper(family = env_family, env_name = env_name,agent_ids=agent_ids,**kwargs)
    elif env_type == 'smaclite':
        env = SMACliteWrapper(map_name=env_name,agent_ids=agent_ids,**kwargs)
    
    return env
def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.tensor(norms), d)
    return total_norm_d
def soft_update(target_net, critic_net, polyak):
        for target_param, param in zip(target_net.parameters(), critic_net.parameters()):
            target_param.data.copy_(polyak * param.data + (1.0 - polyak) * target_param.data)

def get_coma_critic_input_dim(env):
    critic_input_dim = 0
    critic_input_dim = env.get_obs_size() + env.get_state_size() +  (env.n_agents - 1)*env.get_action_size()
    return critic_input_dim
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
    ## what if we periodically empty the replay buffer
    args = tyro.cli(Args)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ## import the environment 
    kwargs = {} #{"render_mode":'human',"shared_reward":False}
    ## Create the pipes to communicate between the main process (COMA algorithm) and child processes (envs)
    conns = [Pipe() for _ in range(args.batch_size)]
    coma_conns, env_conns = zip(*conns)
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
    coma_conns[0].send(("get_env_info", None))
    env_info = coma_conns[0].recv()
    
    
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
    critic_input_dim = get_coma_critic_input_dim(eval_env)
    critic = Critic(
        input_dim=critic_input_dim,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        output_dim=eval_env.get_action_size(),
        num_agents =  eval_env.n_agents
    )
    target_critic = Critic(
        input_dim=critic_input_dim,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        output_dim=eval_env.get_action_size(),
        num_agents =  eval_env.n_agents
    )
    soft_update(
            target_net=target_critic,
            critic_net=critic,
            polyak=1.0
        )
    Optimizer = getattr(optim, args.optimizer) 
    actor_optimizer = Optimizer(actor.parameters(),lr = args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(),lr = args.learning_rate_critic)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    writer = SummaryWriter(f"runs/COMA-{run_name}")
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
    total_nepisodes = 0
    training_step = 0
    while step < args.total_timesteps:
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction, training_step)
        episodes = [{"obs": [],"actions":[],"reward":[],"states":[],"done":[],"avail_actions":[]}
                for _ in range(args.batch_size)]
        
        for coma_conn in coma_conns:
            coma_conn.send(("reset",None))
        
        contents = [coma_conn.recv() for coma_conn in coma_conns]
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
                actions = actor.act(obs,eps=epsilon,avail_action=avail_action)
            for i,j in enumerate(alive_envs):
                coma_conns[j].send(("step",actions[i]))
            
            contents = [coma_conns[i].recv() for i in alive_envs]

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
        
        
        total_nepisodes += args.batch_size
        ## logging
        writer.add_scalar("rollout/ep_reward", np.mean(ep_reward), step)
        writer.add_scalar("rollout/ep_length",np.mean(ep_length),step)
        if args.env_type == 'smaclite':
            writer.add_scalar("rollout/battle_won",np.mean(np.mean([info["battle_won"] for info in ep_stat])), step)
        b_obs,b_actions,b_reward,b_states,b_avail_actions,b_done,b_mask = rb.get_batch()
        ### 1. Compute TD(λ) from "Reconciling λ-Returns with Experience Replay"(https://arxiv.org/pdf/1810.09967 Equation 3)
        with torch.no_grad():
            return_lambda = torch.zeros_like(b_actions).float()
            for ep_idx in range(return_lambda.size(0)):
                ep_len = b_mask[ep_idx].sum()
                last_return_lambda = 0
                for t in reversed(range(ep_len)):
                    if t == (ep_len -1):
                        next_action_value = 0
                    else:
                        next_action_value = target_critic(state = b_states[ep_idx,t+1],
                                                        observations = b_obs[ep_idx,t+1],
                                                        actions =b_actions[ep_idx,t+1],
                                                        avail_actions= b_avail_actions[ep_idx,t+1] )
                        
                        next_action_value = torch.gather(next_action_value,dim=-1, index=b_actions[ep_idx,t+1].unsqueeze(-1)).squeeze()
                        # next_action_value, _ = next_action_value.max(dim=-1)
                        
                        # print(last_return_lambda)
                    return_lambda[ep_idx,t] =last_return_lambda = b_reward[ep_idx,t] + args.gamma * (args.td_lambda * last_return_lambda + (1-args.td_lambda)*next_action_value)
        # with torch.no_grad():
        #     return_lambda = torch.zeros_like(b_actions).float()
        #     for ep_idx in range(return_lambda.size(0)):
        #         ep_len = b_mask[ep_idx].sum()
        #         for t in range(ep_len):
        #             if t < (ep_len - args.nsteps):
        #                 return_t_n =  b_reward[ep_idx,t:t+args.nsteps]
        #                 discounts = torch.tensor([args.gamma ** i for i in range(return_t_n.size(-1))])
        #                 return_t_n = (return_t_n*discounts).sum(-1)
        #                 action_value_t_n = target_critic(state = b_states[ep_idx,t+args.nsteps],
        #                                                 observations = b_obs[ep_idx,t+args.nsteps],
        #                                                 actions =b_actions[ep_idx,t+args.nsteps],
        #                                                 avail_actions= b_avail_actions[ep_idx,t+args.nsteps] )
        #                 # print("action_value_t_n",action_value_t_n.shape)
        #                 # print("b_actions[ep_idx,t+args.nsteps]",b_actions[ep_idx,t+args.nsteps].shape)
        #                 action_value_t_n = torch.gather(action_value_t_n.squeeze(),dim=-1, index=b_actions[ep_idx,t+args.nsteps].unsqueeze(-1))
        #                 return_t_n = return_t_n + args.gamma ** args.nsteps * action_value_t_n
                        
        #             else:
        #                 return_t_n =  b_reward[ep_idx,t:]
        #                 discounts = torch.tensor([args.gamma ** i for i in range(return_t_n.size(-1))])
        #                 return_t_n = (return_t_n*discounts).sum(-1)
        #                 return_t_n = return_t_n.expand(eval_env.n_agents)
        #             # print("return_t_n",return_t_n.shape)
        #             # print("return_lambda[ep_idx,t]",return_lambda[ep_idx,t].shape)
        #             # print("return_t_n",return_t_n.shape)
        #             # print("return_lambda[ep_idx,t]",return_lambda[ep_idx,t].shape)
        #             return_lambda[ep_idx,t] = return_t_n.squeeze()



        ### 2. Update the critic
        # q_values = torch.zeros((*b_obs.shape[:-1],env.get_action_size()))
        critic_losses = []
        ciritc_gradients = 0
        cr_loss = 0
        for t in range(b_obs.size(1)):
            b_q_values = critic(state = b_states[:,t],
                              observations = b_obs[:,t],
                              actions =b_actions[:,t])
            # q_values[:,t] = b_q_values.clone().detach()
            b_q_values = torch.gather(b_q_values,dim=-1, index=b_actions[:,t].unsqueeze(-1)).squeeze()
            q_targets = return_lambda[:,t]
            
            critic_loss = F.smooth_l1_loss(b_q_values[b_mask[:,t]],q_targets[b_mask[:,t]])
            

            critic_losses.append(critic_loss.item())
            cr_loss += critic_loss * b_mask[:,t].sum()
        critic_optimizer.zero_grad()
        cr_loss = cr_loss / b_mask.sum()
        cr_loss.backward()
        grads = [p.grad for p in critic.parameters() ]
        grad_norm_2 = norm_d(grads,2)
        ciritc_gradients += grad_norm_2
        critic_optimizer.step()

        ### 3. Update actor
        actor_losses = []
        entropies = []
        gradients = 0
        ac_loss = 0
        for t in range(b_obs.size(1)):
            pi = actor.logits( b_obs[:,t],eps=epsilon,avail_action=b_avail_actions[:,t]) 
            # pi = F.softmax(logits,dim=-1)
            log_pi = torch.log(pi + 1e-8)
            entropy_loss = -(pi*log_pi).sum(dim=-1)[b_mask[:,t]]
            entropy_loss = entropy_loss.sum() #/ b_mask[:,t].sum()
            entropies.append(entropy_loss.item()/ b_mask[:,t].sum())
            
            q_values = critic(state = b_states[:,t],
                              observations = b_obs[:,t],
                              actions =b_actions[:,t])
            q_values = q_values.clone().detach()
            coma_baseline = (pi * q_values)
            coma_baseline = coma_baseline.sum(dim = -1)
            current_q =  torch.gather(q_values,dim=-1, index=b_actions[:,t].unsqueeze(-1)).squeeze()
            advantage = (current_q - coma_baseline).detach()
            if b_actions[:,t].sum() > eval_env.n_agents:
                advantage = (advantage - advantage[b_mask[:,t]].mean())/(advantage[b_mask[:,t]].std() + 1e-8)
            log_pi = torch.gather(log_pi,dim=-1, index=b_actions[:,t].unsqueeze(-1)).squeeze()
            actor_loss = (log_pi[b_mask[:,t]] * advantage[b_mask[:,t]]).sum()
            # actor_loss /= (b_mask[:,t].sum())
            # actor_losses.append(actor_loss.item())
            actor_loss = - actor_loss - args.entropy_coef*entropy_loss
            ac_loss += actor_loss
        actor_optimizer.zero_grad()
        ac_loss = ac_loss / b_mask.sum()
        ac_loss.backward()
        grads = [p.grad for p in actor.parameters() ]
        grad_norm_2 = norm_d(grads,2)
        gradients += grad_norm_2
        actor_optimizer.step()
        # writer.add_scalar("train/critc_loss", np.mean(critic_losses), step)
        training_step+= 1
        if training_step % args.target_network_update_freq == 0:
            soft_update(
                target_net=target_critic,
                critic_net=critic,
                polyak=args.polyak
                )
        writer.add_scalar("train/critc_loss", cr_loss, step)
        writer.add_scalar("train/epsilon", epsilon, step)
        writer.add_scalar("train/train_steps", training_step, step)
        # writer.add_scalar("train/actor_loss", np.mean(actor_losses), step)
        writer.add_scalar("train/actor_loss", ac_loss.item(), step)
        writer.add_scalar("train/entropy", np.mean(entropies), step)
        writer.add_scalar("train/actor_gradients", gradients /b_obs.size(1), step)
        writer.add_scalar("train/critic_gradients", ciritc_gradients /b_obs.size(1), step)

        if (total_nepisodes/args.batch_size) % args.eval_steps == 0:
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
                    actions = actor.act(eval_obs,avail_action=mask_eval)
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
                



        
        

