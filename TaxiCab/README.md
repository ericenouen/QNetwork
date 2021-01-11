# Taxi Cab OpenAI Gym implementation.

## Q-Learning
Since the TaxiCab problem only has 500 discrete states and 6 deterministic actions, we can use a Q-Table to represent each state-action pair in order to learn what the optimal action would be within each state.

## Epsilon-Greedy
On each time step t, the agent will either perform a random action with probability epsilon, or simply perform the action that maximizes the reward from the Q-table.
In this task, I decay epsilon over time so that the agent can perform greedily as it learns more.

### Update Step
To update this Q-table, we need to use the update rule for Q-learning. This rule is shown below:

![Q-update](https://user-images.githubusercontent.com/54828661/104230085-15b58580-541b-11eb-9700-8f2e7a81dc27.jpg)

At each time step t this equation is updating the Q-value for a specific state-action pair.

Essentially, this equation is taking small steps towards the estimated target (reward + estimate of future discounted reward). This estimated target takes the reward received after performing an action in the given time step, and then uses bootstrapping to estimate the future discounted reward by maximizing over all actions for the Q-values of the next state. This is an estimate of the actual value function of the next state, which would show the impact of an action on future states. This forces the algorithm to not only maximize current rewards, but ensure that it does not get stuck in nonoptimal states in the future that prevent them from receiving reward. The discount factor ensures that future states are worth slightly less reward than current states, and pushes the agent to acknowledge time. I found that for this specific task, the discount factor is not as important and can be many varying values and still converge.

The old q value is the starting point of the equation, but it is subtracted by alpha times the old q value. Then (1 - alpha) times the old q value is added with alpha times the value of the TD target. This is moving the old q value in the direction of the target by taking parts of each using the learning rate alpha. With continuous updates, this reward signal is able to teach the Q-table which are the best actions to take in each state, and the agent is able to perform the task very well.

## Performance
When I allow epsilon to drop all the way to zero, I am able to produce this graph of the 100-timestep average reward, which is producing an average reward of approximately 8 by the end of training.
