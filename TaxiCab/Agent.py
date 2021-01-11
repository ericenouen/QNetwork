import numpy as np

class AgentQ(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_values = np.zeros([500, 6]) # 500 states, 6 actions: Q(s,a)
        
        # Hyperparameters
        self.alpha = 0.01
        self.decay = 0.99
        self.epsilon = 0.5
        self.epsilon_decay = 0.95

    
    def act(self, state, reward, done):
        self.epsilon = max(0.05, self.epsilon * self.epsilon_decay) # Decay epsilon if above 0.5
        # Use epsilon greedy
        if np.random.random_sample() < self.epsilon:
            return self.action_space.sample() # Return random action
        else:
            values = self.q_values[state, :] # Return the action that corresponds to the max q value
            return np.argmax(values)

    def update_qvalues(self, state, reward, next_state, action, done):
        # Update Step for the network
        self.q_values[state, action] += self.alpha * (reward + self.decay * np.max(self.q_values[next_state, :]) - self.q_values[state, action])