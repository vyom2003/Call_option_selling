import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import math
import matplotlib.pyplot as plt

# Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

np.random.seed(0)
torch.manual_seed(0)

def get_state(curr_state, drift, days, action):
    """
    Calculate the next state based on the current state and drift.
    """
    if action==1:
        return 2*drift*days+1
    delta = np.random.randint(-drift, drift, dtype=int)
    return max(0, min(curr_state + delta, 2 * days * drift))

def get_reward(curr_state, action, days, drift, start_price, strike_price):
    """
    Calculate the reward based on the current state and action.
    """
    if action == 0:
        return 0
    else:
        return curr_state - days * drift + start_price - strike_price

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-8):
        super(SharedAdam, self).__init__(params, lr=lr)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, num_hidden_layers,hidden_size, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.states = []
        self.actions = []
        self.rewards = []
        self.done=[]
        self.input_layer = nn.Linear(input_dims, hidden_size)
        self.input_activation = nn.LeakyReLU()

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(int(num_hidden_layers)):
            layer = nn.Linear(hidden_size, hidden_size)
            self.hidden_layers.append(layer)
            self.hidden_layers.append(nn.LeakyReLU())

        # Output layer
        self.output_layer_value = nn.Linear(hidden_size, 1)
        self.output_activation_value = nn.Identity()

        self.output_layer_policy = nn.Linear(hidden_size,n_actions)
    
    def remember(self, state, action , reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.done.append(done)
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.done=[]
    
    def forward(self, state):
        x = self.input_layer(state)
        x = self.input_activation(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        value = self.output_layer_value(x)
        value = self.output_activation_value(value)

        policy = self.output_layer_policy(x)
        return policy, value
    
    def calc_R(self, done):
        states = torch.tensor(self.states, dtype=torch.float32)
        _, v = self.forward(states)

        # R = v[-1]*(1-int(done))

        batch_return = []
        for i in range(len(self.states)):
            if(self.done[i]):
                batch_return.append(self.rewards[i])
            else:
                batch_return.append(self.rewards[i] + self.gamma*v[i])
            
        # for reward in self.rewards[::-1]:
        #     R= reward  + self.gamma*R
        #     batch_return.append(R)
        # batch_return.reverse()
        return torch.tensor(batch_return, dtype=torch.float32)
    
    def calc_loss(self, done):
        states = torch.tensor(self.states, dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.float32)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()

        critic_loss = (returns - values)**2
        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns - values)

        total_loss = (critic_loss + actor_loss).mean()
        return total_loss

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float32)
        pi, _ = self.forward(state)
        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        return action

T = 5  # Number of time steps
drift = 5
start_price = 500
strike_price = 505
days=T
T_MAX=5
hidden_size =64
num_hidden_layers=4

class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, gamma, name, global_ep_idx, num_iter=2000, start_states=None, reward_arr=None):
        super(Agent, self).__init__()

        self.local_actor_critic = ActorCritic(input_dims = input_dims, n_actions=n_actions, gamma=gamma, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        self.global_actor_critic = global_actor_critic

        self.name= 'w%02i'% name
        self.episode_idx = global_ep_idx
        self.optimizer = optimizer
        self.num_iter=num_iter
        self.start_states=start_states
        self.reward_arr=reward_arr
    
    def run(self):
        t_step =1
        while self.episode_idx.value < self.num_iter:
            curr_state = np.random.randint(0,2*T*drift+1)
            time = np.random.randint(0,T)
            done=False
            reward_curr = 0
            t0=time
            start_state=curr_state
                
            while not done:
                action = self.local_actor_critic.choose_action(np.array([curr_state,time]))
                next_state = get_state(curr_state, drift, days, action)
                reward = get_reward(curr_state, action, days, drift, start_price, strike_price)
                reward_curr += reward * math.pow(self.local_actor_critic.gamma, time - t0)
                self.local_actor_critic.remember(np.array([curr_state,time]), action, reward, done)
                if time == days or action == 1:
                    done = True

                if t_step% T_MAX ==0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(self.local_actor_critic.parameters(), self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                t_step+=1
                curr_state =next_state
                time+=1
            
            with self.episode_idx.get_lock():
                self.episode_idx.value +=1
                self.reward_arr.append((self.episode_idx.value,reward_curr))
                x = (self.episode_idx.value,start_state, t0)
                self.start_states.append(x)
                if(self.episode_idx.value%10)==0:
                    print_policy(self.global_actor_critic,drift, start_price, T, episode=self.episode_idx.value)
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % reward, 'loss ', loss.item())

def print_policy(model, drift, start_price, T, episode=0):
    policy = []  # Dictionary to store policy for each time-price pair

    for time in range(T):
        time_policy = []  # Policy for current time
        for price in range(2 * T * drift + 1):
            # Move input data to GPU
            state = torch.tensor([price, time], dtype=torch.float32)
            action_probs,_ = model(state.unsqueeze(0))
            action_probs = torch.softmax(action_probs, dim=1)
            time_policy.append(action_probs.argmax().item())
        policy.append(time_policy)

    output_file='policy_output.txt'
    with open(output_file, "a") as f:
        f.write(f"Episode : {episode}\n")
    # Write policy for each time-price pair to the file
        for time, price_policy in enumerate(policy):
            f.write(f"Time: {time + 1}\n")
            f.write(str(price_policy) + "\n")

        optimal_policy=[]
        for time,price_policy in enumerate(policy):
            for i,price in enumerate(price_policy):
                if(price==1):
                    optimal_policy.append(i-T*drift+start_price)
                    f.write(f"Time: {time + 1}, Price:{i-T*drift+start_price} \n")
                    break
    
    return optimal_policy

def transition_probability(drift):
    """
    Calculates the transition probability for the given drift.
    """
    return 1 / (2 * drift + 1) 

def value_iteration(gamma=0.99):
    """
    Performs value iteration to find the optimal policy.
    """
    start_price, strike_price = 500, 505
    drift = 5  # Change in stock price on each day
    T = 5  # Number of days till expiry

    V = np.zeros((2 * T * drift + 1, T))
    policy = np.zeros((T, 2 * T * drift + 1))

    # Initialize the value function and policy
    for state in range(2 * T * drift + 1):
        V[state][T - 1] = max(0, state - (strike_price - start_price) - T * drift)
        if V[state][T - 1] > 0:
            policy[T - 1][state] = 1

    # Perform value iteration
    for t in reversed(range(T - 1)):
        Q = np.zeros((2 * T * drift + 1, 2))
        for state in range(2 * T * drift + 1):
            for action in [0, 1]:
                new_value = 0.0
                if action == 0:
                    reward = 0
                    for next_state in range(state - drift, state + drift):
                        if 0 <= next_state <= 2 * T * drift:
                            new_value += transition_probability(drift) * (reward + gamma * V[next_state][t + 1])
                else:
                    reward = state - (strike_price - start_price) - T * drift
                    new_value = reward
                Q[state][action] = new_value
            V[state][t] = max(Q[state])
            # Update policy based on Q values
            if Q[state][0] >= Q[state][1]:
                policy[t][state] = 0
            else:
                policy[t][state] = 1
    optimal = []
    for t in range(T):
        for i in reversed(range(2 * T * drift + 1)):
            if policy[t][i] == 0:
                optimal.append(i - T * drift + start_price +1)
                break
    plt.plot(optimal, marker='o')
    plt.title('Optimal Policy: Threshold Price vs Days (By Value Iteration)')
    plt.xlabel('Days')
    plt.ylabel('Threshold Price')
    plt.grid(True)  # Add grid
    plt.show()
    return V

if __name__ == '__main__':
    lr=2*1e-6
    n_actions=2
    input_dims=2
    num_iter=1395
    start_states=mp.Manager().list()
    reward_arr=mp.Manager().list()
    global_actor_critic = ActorCritic(input_dims=input_dims, n_actions=n_actions,num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr)
    global_ep =mp.Value('i', 0)

    workers = [Agent(global_actor_critic=global_actor_critic,optimizer=optim,input_dims=input_dims,n_actions= n_actions, gamma=0.99, name =i, global_ep_idx=global_ep, num_iter=num_iter, start_states=start_states, reward_arr=reward_arr) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]
    
    optimal_policy = print_policy(global_actor_critic,drift,start_price, T)

    plt.plot(optimal_policy, marker='o')  # Plotting every 100th value
    plt.title('Optimal Policy: Threshold Price vs Days (By A3C)')
    plt.xlabel('Days')
    plt.ylabel('Threshold Price')
    plt.grid(True)  # Add grid
    plt.savefig('A3C_policy')
    plt.show()

    V = value_iteration()

    
    start_states = sorted(start_states, key=lambda x: x[0])
    start_states = [(t[1], t[2]) for t in start_states]

    reward_arr = sorted(reward_arr,key= lambda x:x[0])
    reward_arr = [t[1] for t in reward_arr]

    regrets=[]
    for i in range(0,len(start_states)):
        regrets.append(V[start_states[i]] - reward_arr[i])
    
    arr_tot = []
    for i in range(len(regrets)):
        if(i==0):
            arr_tot.append(regrets[i])
        else:
            arr_tot.append(arr_tot[-1]+regrets[i])

    plt.figure(figsize=(10,6))
    plt.plot(arr_tot)
    plt.title("Cumulative Regret vs Episodes")
    plt.savefig('A3C_cum_regret')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(regrets)
    plt.title("Instantaneous Regret vs Episodes")
    plt.savefig('A3C_inst_reg')
    plt.show()


