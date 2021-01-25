import Agent

import argparse
import sys

import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Create environment and agent
    env = gym.make("Taxi-v3")
    agent = Agent.AgentQ(env.action_space)

    # Initialize variables to store game progress and performance
    episode_count = 50000
    reward = 0
    done = False

    steps = 0
    episode_reward = 0
    rewardList = []
    average_100 = []
    success = 0

    for i in range(episode_count):
        # Reset the environment and episodic reward at the start of each episode
        state = env.reset()
        episode_reward = 0
        while True:
            steps += 1

            action = agent.act(state, reward, done) # Grab action from agent
            next_state, reward, done, _ = env.step(action) # Take action in environment

            agent.update_qvalues(state, reward, next_state, action, done) # Update the Q-values
            
            # Update state and episodic reward
            state = next_state
            episode_reward += reward

            # env.render()
            if done:
                if episode_reward >= 10:
                    # Measure the amount of times the agent correctly drops off the passenger in less than 10 time steps
                    success += 1
                rewardList.append(episode_reward)
                first = max(i-100, 0)
                
                # Store 100-episode averages
                average_100.append(sum(rewardList[first:i]) / (i - first + 1)) # Should be first:i+1

                if (i+1) % 25 == 0:
                    # Print the episode number every 25 episodes
                    print("Episodes: " + str(i+1))
                    print(average_100[i])
                break
    print("Times: " + str(success))
    # Plot the 100-episode averages
    plt.plot(range(episode_count), average_100)
    plt.show()
    env.close()
