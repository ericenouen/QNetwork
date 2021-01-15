import numpy as np
import Network
import tensorflow as tf

class AgentQ(object):
    def __init__(self, action_space):
        self.action_space = action_space

        self.online_network = Network.QNetwork() # Network used for learning and to take actions
        self.target_network = Network.QNetwork() # Network used in the Q-learning update step for the target
        self.copy_target_weights()

        self.experience_replay = []
        self.num_actions = 2
        self.replay_size = 100000
        
        self.discount = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.05
        self.epsilon_delta = (self.epsilon - self.epsilon_min) / 200000

    def copy_target_weights(self):
        # Copy the online network's weights to the target network
        print("Copy Weights")
        weights = self.online_network.get_weights()
        self.target_network.set_weights(weights)
    
    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_delta
        # Return the action that corresponds to the max q value from the network
        # Use epsilon greedy
        if np.random.random_sample() < self.epsilon:
            return self.action_space.sample()
        else:
            action = self.online_network.eval(np.expand_dims(state, 0))
            return action.item(0)

    def add_experience(self, state, reward, action, next_state, done):
        # Add an experience to the replay buffer
        experience = {"state": state, "reward":reward, "action":action, "next_state":next_state, "done":done}
        if len(self.experience_replay) >= self.replay_size:
            self.experience_replay.pop(0)
        self.experience_replay.append(experience)

    def update_network(self):
        batch = []
        batch_size = 256

        # Randomly pick experiences from replay buffer
        for _ in range(batch_size):
            idx = int(np.random.randint(0, len(self.experience_replay)))
            batch.append(self.experience_replay[idx])

        # Grab individual components of the experience
        inputs = np.reshape([b["state"] for b in batch], (batch_size,4))
        actions = np.array([b["action"] for b in batch])
        rewards = np.array([b["reward"] for b in batch])
        next_inputs = np.array([b["next_state"] for b in batch])
        done_values = np.array([b["done"] for b in batch])

        actions_one_hot = np.eye(self.num_actions)[actions] # Action left or right, one-hot-encoded
        next_qvalues = np.squeeze(self.target_network.network.predict(next_inputs)) # Use target network to predict next q values
        expected_future_return = np.amax(next_qvalues, axis=-1) # Use the maximum q value from the next state

        # Use the done values to remove future reward if it causes the episode to end
        expected_future_return = expected_future_return * ~done_values

        # ~done_values results in True values zeroing out the expected_future_return
        targets = rewards + self.discount * expected_future_return

        self.online_network.update(inputs, targets, actions_one_hot)


    def save_weights(self):
        # Save weights to a file
        self.online_network.network.save("DeepQCartPole/model/weights")


    def load(self):
        # Load the weights from a file into the target network
        self.online_network.network = tf.keras.models.load_model("DeepQCartPole/model/weights")