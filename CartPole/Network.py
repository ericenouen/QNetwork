import tensorflow as tf
import numpy as np

class QNetwork(object):
    def __init__(self):
        self.network = self.model()
        self.learning_rate = 0.0001

    def model(self):
        # Input the state as 4 different values
        # Output the Q-values for the two actions
        inputs = tf.keras.Input(shape=(4,), dtype=tf.float64)
        x = tf.keras.layers.Dense(8, activation='relu')(inputs)
        x = tf.keras.layers.Dense(8, activation='relu')(x)
        outputs = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def update(self, inputs, targets, actions_one_hot):
        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate) # Create an Adam optimizer with the given learning rate

        with tf.GradientTape() as tape:
            qvalues = tf.squeeze(self.network(inputs)) # Grab the Q-values from this state
            preds = tf.reduce_sum(qvalues * actions_one_hot, axis=1) # Only select the Q-value associated with the action taken
            loss = tf.losses.mean_squared_error(targets, preds) # Find the mean squared error between the predicted Q value and target
        
        gradients = tape.gradient(loss, self.network.trainable_weights) # Calculate the gradients w.r.t the weights of the network
        optimizer.apply_gradients(zip(gradients, self.network.trainable_weights)) # Apply the gradients found to the weights
        

    def set_weights(self, weights):
        # Set the weights of the network
        self.network.set_weights(weights)

    def get_weights(self):
        # Get the weights of the network
        return self.network.get_weights()
    
    def eval(self, state):
        # Return the optimal action for the given state
        q_values = self.network.predict(state)
        return np.squeeze(np.argmax(q_values, axis=-1))