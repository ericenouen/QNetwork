import Agent

import argparse
import sys

import gym
from gym import wrappers, logger
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


if __name__ == '__main__':

    tf.enable_eager_execution()
    # Create environment and agent
    env = gym.make("CartPole-v0")
    agent = Agent.AgentQ(env.action_space)

    # Set up parameters for playing the game
    episode_count = 50000
    reward = 0
    done = False
    
    steps = 0
    episode_reward = 0
    rewardList = []
    average_100 = []
    success = 0
    best_agent = 150

    for i in range(episode_count):
        # Reset state and episode reward on each new episode
        state = env.reset()
        episode_reward = 0
        while True:
            steps += 1

            action = agent.act(state) # Get action from agent
            next_state, reward, done, _ = env.step(action) # Take action in environment

            agent.add_experience(state, reward, action, next_state, done) # Add experience to replay buffer
            if len(agent.experience_replay) >= 1000:
                # Update online network once replay buffer has over 1000 samples
                agent.update_network()

            state = next_state
            episode_reward += reward

            # env.render()
            if done:
                if steps % 10000 == 0:
                    # Copy target network weights from online network every 10000 steps
                    agent.copy_target_weights()
                
                if episode_reward >= 180:
                    # Count how many times agent receives >= 180 reward
                    success += 1
                
                # Add up the episodic rewards and calculate the 100-episode average
                rewardList.append(episode_reward)
                first = max(i-100, 0)
                average_100.append(sum(rewardList[first:i]) / (i - first + 1)) # Should be first:i+1

                if average_100[i] >= best_agent:
                    # Save the weights of the network if it performs well enough
                    best_agent = average_100[i]

                    # Considered solved when the average return is greater than or equal to
                    # 195.0 over 100 consecutive trials.
                    agent.save_weights()

                if (i+1) % 25 == 0:
                    # Print out episode number and average reward every 25 episodes
                    print("Episodes: " + str(i+1))
                    print(average_100[i])
                break

    print("Times: " + str(success))
    # Plot the average reward
    plt.plot(range(episode_count), average_100)
    plt.show()
    env.close()
