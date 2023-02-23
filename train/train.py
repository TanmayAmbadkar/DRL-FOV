import time

import matplotlib.pyplot as plt


def train(agent, env, episodes=100):

    observation = env.reset()
    batch_size = 16
    for _ in range(episodes):
        done = False
        while not done:
            action = agent.training_action(observation)
            next_observation, reward, done = env.step(action)
            agent.memory.append((observation, action, reward, next_observation, done))
            observation = next_observation
            agent.batch_train(batch_size)

    return agent
