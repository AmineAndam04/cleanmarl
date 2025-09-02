import torch
import tyro
import datetime
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from env.pettingzoo_wrapper import PettingZooWrapper
from env.smaclite_wrapper import SMACliteWrapper
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    env_type: str = "smaclite" #"pz"
    """ Pettingzoo, SMAClite ... """
    env_name: str = "MMM" #"simple_spread_v3" #"pursuit_v4"
    """ Name of the environment """
    env_family: str ="mpe"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    batch_size: int = 30
    """ Number of episodes to collect each rollout"""
    actor_hidden_dim: int = 64
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 2
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 64
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 2
    """ Number of hidden layers of critic network"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate: float =  0.0003
    """ Learning rate"""
    total_timesteps: int = 2000000
    """ Total steps in the environment during training"""


class Actor(nn.Module):
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
    
class Critic(nn.Module):
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
    if isinstance(env,PettingZooWrapper): # we don't have a true environment state.
        critic_input_dim += env.get_obs_size()
    elif isinstance(env,SMACliteWrapper):
        critic_input_dim += env.get_obs_size() + env.get_state_size()
    else:
        raise NotImplementedError(f"get_coma_critic_input_dim not implemented for type {type(env).__name__}")
    critic_input_dim += env.get_action_size() - 1
    return critic_input_dim


if __name__ == "__main__":
    ## what if we periodically empty the replay buffer
    args = tyro.cli(Args)
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
    
    ## Initialize the actor, critic and target-critic networks
    actor = Actor(
        input_dim=env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=env.get_action_size()
    )
    critic_input_dim = get_coma_critic_input_dim(env)
    critic = Critic(
        input_dim=critic_input_dim,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        output_dim=env.get_action_size()
    )
    target_critic = Critic(
        input_dim=critic_input_dim,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        output_dim=env.get_action_size()
    )
    soft_update(
            target_net=target_critic,
            critic_net=critic,
            polyak=1.0
        )
    Optimizer = getattr(optim, args.optimizer) 
    actor_optimizer = Optimizer(actor.parameters(),lr = args.learning_rate)
    critic_optimizer = Optimizer(critic.parameters(),lr = args.learning_rate)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    writer = SummaryWriter(f"runs/COMA-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    num_episode = 0
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    step = 0
    while step < args.total_timesteps:
        ## Create the rollout buffer
        