# we first define deep Q-networks, experience replay buffer and then we define a DQN agent which incorporates both
# q-networks and reply buffer


import numpy as np
import torch
from torch import nn
from collections import namedtuple, deque
from copy import deepcopy
from PIL import Image


# Deep Q-networks
# noinspection PyArgumentList
class QNetworks(nn.Module):
    """
    Define Q-network that takes observations as inputs and outputs Q values for all actions
    """
    def __init__(self, env, learning_rate):
        # the input of the Q networks is state and the output is the action
        super().__init__()
        self.nn_inputs = env.observation_space.shape[0]
        self.nn_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        # set up the neural networks
        self.network = nn.Sequential(
            nn.Linear(self.nn_inputs, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, self.nn_outputs, bias=True)
        )
        # optimizer: using adam
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # for a state s, get Q(s,a) for all actions
    def get_qvalue(self, state):
        # reconstruct state as an array
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        # use cpu for computing the output of the Qnetwork
        state_t = torch.FloatTensor(state)
        return self.network(state_t)

    # get the greedy action based on Q values
    def greedy_action(self, state):
        qvalue = self.get_qvalue(state)
        action = torch.max(qvalue, dim=-1)[1].item()
        return action

    # epsilon-greedy policy
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.greedy_action(state)
        return action


# experience replay
class ReplayBuffer:
    """
    Experience replay buffer that is base on deque and stores tuples (s_t,a_t,r_t,done,s_{t+1})
    memory_size: total samples
    initial: #initial samples size
    """
    def __init__(self, memory_size=10000, initial=2000):
        self.memory_size = memory_size
        # we begin sampling after initial samples are recorded
        self.initial = initial
        self.Buffer = namedtuple('Buffer',
                                 field_names=['state', 'action', 'reward', 'done', 'next_state'])
        # deque: when memory bank is full, we throw away earlier samples
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size,
                                   replace=False)
        # Use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(
            self.Buffer(state, action, reward, done, next_state))

    def ready(self):
        if len(self.replay_memory) > self.initial:
            return True
        else:
            return False


# DQN agent
# noinspection PyArgumentList
class DQNagent:
    def __init__(self, env, network, buffer, gamma=0.99, epsilon=0.05, batch_size=32,
                 target_network_update=400, network_update=4):
        # basic parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.step_count = 0
        # environment
        self.env = env
        self.s0 = self.env.reset()
        # training related
        self.network = network
        self.target_network = deepcopy(network)
        self.rewards = 0
        self.buffer = buffer
        # Q network update frequency
        self.update_freq = network_update
        # target network update frequency
        self.copy_freq = target_network_update
        # pre-allocation for date recording
        self.training_rewards = []
        self.mean_training_rewards = []
        # for computing the average
        self.window = 100
        # average reward when cartpole is solved
        self.reward_thresh = 195

    def explore_mode(self):
        action = self.env.action_space.sample()
        s1, r, done, _ = self.env.step(action)
        self.buffer.append(self.s0, action, r, done, s1)
        self.s0 = s1.copy()
        if done:
            self.s0 = self.env.reset()

    def train_mode(self):
        action = self.network.get_action(self.s0, epsilon=self.epsilon)
        self.step_count += 1
        s1, r, done, _ = self.env.step(action)
        self.rewards += r
        self.buffer.append(self.s0, action, r, done, s1)
        self.s0 = s1.copy()
        return done

    def display_mode(self, filename):
        frames = []
        while True:
            action = self.network.greedy_action(self.s0)
            self.env.render()
            frames.append(Image.fromarray(self.env.render(mode='rgb_array')))
            s1, r, done, _ = self.env.step(action)
            self.s0 = s1.copy()
            if done:
                break
        self.env.close()
        im = Image.new('RGB', frames[0].size)
        im.save(filename, save_all=True, append_images=frames)

    def dqn_minloss(self, ddqn):
        # first get a batch
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        # unpack state-action-reward tuple
        states, actions, rewards, dones, next_states = [i for i in batch]
        Rewards = torch.tensor(rewards).reshape(-1, 1)
        Actions = torch.tensor(actions).reshape(-1, 1)
        Dones = torch.ByteTensor(dones)
        # get  values
        qvalues = torch.gather(self.network.get_qvalue(states), 1, Actions)
        qvalues = torch.reshape(qvalues, (-1, 1))
        # max Q(s,a)
        if ddqn:  # double q learning
            # get actions by doing argmax Q
            max_action = torch.max(self.network.get_qvalue(next_states), dim=-1)[1].detach()
            max_action = torch.reshape(max_action, (-1, 1))
            # evaluate the argmax
            target_qvalues = self.target_network.get_qvalue(next_states)
            qmax = torch.gather(target_qvalues, 1, max_action).detach()
        else:  # vanilla q learning
            qmax = torch.max(self.target_network.get_qvalue(next_states), dim=-1)[0].detach()
            qmax = torch.reshape(qmax, (-1, 1))

        qmax[Dones] = 0  # terminal state
        label_qvalues = self.gamma * qmax + Rewards
        mse = nn.MSELoss()
        loss = mse(qvalues, label_qvalues)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

    def dqn_train(self, ddqn=False, max_episode=10000, test=100, rate=0.8):
        """
        deep Q-learning algorithm
        ddqn: if True, then perform double Q-learning, else vanilla deep Q learning
        max_episode: upper limit for #episodes
        test: #episodes that are inspected to see whether episodic rewards are above the threshold
        rate: required success rate for latest consecutive episodes
        """
        run = True
        eps = 0
        while run:
            # if buffer is ready
            if self.buffer.ready():
                self.s0 = self.env.reset()
                # while not solved
                while True:
                    done = self.train_mode()
                    # if one episode terminates
                    if done:
                        # one episode is complete
                        eps += 1
                        # record rewards and mean rewards
                        self.training_rewards.append(self.rewards)
                        mean_rewards = np.mean(self.training_rewards[-self.window:])
                        self.mean_training_rewards.append(mean_rewards)
                        print('Episode ' + str(eps) + '  Mean rewards ' + str(mean_rewards))
                        # reset
                        self.rewards = 0
                        self.s0 = self.env.reset()
                        if eps > max_episode:  # exceed the upper limit, terminate the training
                            print('the number of episodes exceeds the upper limit')
                            run = False
                            break
                        if mean_rewards > self.reward_thresh:  # if mean is above the threshold
                            if eps > test:  # check the success rate
                                rewards_array = self.training_rewards[-test:]
                                success = [value for value in rewards_array if value > self.reward_thresh]
                                if len(success) / test > rate:
                                    run = False
                                    print('The problem is solved in ' + str(eps) + ' episodes')
                                    break

                    # update Q networks and target networks
                    if self.step_count % self.update_freq == 0:
                        self.dqn_minloss(ddqn)
                    if self.step_count % self.copy_freq == 0:
                        self.target_network.load_state_dict(self.network.state_dict())

            else:
                self.explore_mode()

        return eps, self.training_rewards, self.mean_training_rewards
