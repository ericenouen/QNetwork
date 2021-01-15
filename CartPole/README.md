# CartPole OpenAI Gym implementation.

## Environment
The CartPole environment has the following description:

        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

The state-space is continuous, which makes it a good opportunity to apply a function approximator instead of a table since a table can only map discrete values.

The action space, however is a discrete value, 0 or 1, representing pushing the cart to the left or to the right. If the pole falls below a given threshold, the cart moves off the screen, or the episode length reaches 200, the episode is terminated.

To "solve" this environment, the agent needs to receive an average return greater than or equal to 195.0 over 100 consecutive trials.

## Deep Q-Learning
In the TaxiCab environment, we were able to create a table that could map all discrete state and action pairs together and estimate their Q-values in a tabular format. With this environment, we instead use a simple feed forward neural network to learn a function that will input the state and output the Q-values for the two discrete actions "left" and "right". This allows us to get past the problem of a continous state-space, but we still need a way to apply the update step from the tabular format into a gradient step for the neural network. This will use the same equation, but will instead take gradient steps in the direction of the temporal difference target, which is the reward plus expected discounted future reward.

![Q-update](https://user-images.githubusercontent.com/54828661/104230085-15b58580-541b-11eb-9700-8f2e7a81dc27.jpg)

To implement this, you pull the next Q-values from the network using the next_state and take its maximum value. Then you have to incorporate the done states into the network, by zeroing out any expected future reward whenever the episode ends, because it will not be able to gain any more reward in actuality and the network needs to learn that the end of an episode indicates it can no longer earn more reward which will cause it to try and keep the episode going as long as possible. Then the target is just the rewards added to the expected_future_reward multiplied by a discount factor.

To update the network weights, you run the state inputs through the neural network and only grab the output associated with the action taken in that time step, then calculate the mean squared error between the target and the predictions. Finally, you use an optimizer such as Adam to apply these gradients to the network's weights and update the network.

I used the following FCN as my function approximator:

![Network](https://user-images.githubusercontent.com/54828661/104772412-64c42900-5741-11eb-9b09-13d3b0e352fd.jpg)

There are four state inputs, two hidden layers with eight nodes each, and an output layer for the two discrete actions.

## Hyperparameters
For training, many different hyperparameters had to be selected to ensure the network was able to solve the environment. 
* 50,000 Episodes so the model had the chance to properly learn the function
* Copying weights every 10,000 steps for slower updates to try to avoid divergence
* A replay size of 100,000 to make sure any valuable experience wasn't lost
* A discount factor of .95 to ensure the agent continues to plan for future rewards
* An initial epsilon value of 0.5 to force the agent to explore at the beginning
* A minimum epsilon value of 0.05 or 0.10 to allow the agent to act greedily near the end
* 200,000 steps to go from initial epsilon value to minimum epsilon value, to smoothly become more greedy over time
* A batch size of 256 to allow for smoother updates
* A learning rate of 0.0001 to try and keep the model from diverging

## Results
Minimum Epsilon value of 0.10:

![Figure_1](https://user-images.githubusercontent.com/54828661/104774755-3ba59780-5745-11eb-8416-a66520a6f2a7.png)

![Figure_2](https://user-images.githubusercontent.com/54828661/104774764-3e07f180-5745-11eb-927f-a3ca34513946.png)

Minimum Epsilon value of 0.05:

![Figure_3](https://user-images.githubusercontent.com/54828661/104774774-419b7880-5745-11eb-856b-b93e2933b7a3.png)

![Figure_4](https://user-images.githubusercontent.com/54828661/104774787-452eff80-5745-11eb-9e9e-5ffcdb7020d2.png)

As is clear from the variability of these graphs, DQN as an algorithm can be incredibly unstable and can even catastrophically forget a solution after learning how to solve the environment. This can be remediated by saving the weights for the network when it performs optimally, and by using a small learning rate with a lot of episodes.

An example of the Agent playing can be seen below:


![Video](https://user-images.githubusercontent.com/54828661/104775294-1b2a0d00-5746-11eb-9632-169cae7ef628.gif)
