import matplotlib.pyplot as plt
import numpy as np
import torch

from agent.agent import Agent
from data_loader.data_loader import load_data
from environment.environment import Environment
from params import params
from train.train import train

images_color, images_saliency = load_data(params["images_dir"])

# TRAINING

env = Environment(
    saliency_frames=images_saliency,
    colored_frames=images_color,
    n_frames=params["Environment"]["n_frames"],
    video_frames=params["Environment"]["video_frames"],
    frame_width=params["Environment"]["frame_width"],
    frame_height=params["Environment"]["frame_height"],
)

agent = Agent(n_frames=params["Environment"]["n_frames"])

if params["train"]:
    agent = train(agent, env, episodes=params["Agent"]["episodes"])
    torch.save(agent.model.state_dict(), params["Agent"]["save_path"])

# Testing
agent.model.load_state_dict(torch.load(params["Agent"]["save_path"]))
rendered = np.zeros((env.video_frames, env.frame_height, env.frame_width, 3))
observation = env.reset()
batch_size = 12
for episode in range(1):
    done = False
    count = 0
    reward_episode = []
    while not done:
        rendered[count : count + env.n_frames] = env.render()
        count += env.n_frames
        action = agent.action(observation)
        next_observation, reward, done = env.step(action)
        observation = next_observation
        reward_episode.append(reward)
        if done:
            plt.plot(reward_episode)
            plt.show()

# Rendering video
fig = plt.figure()
viewer = fig.add_subplot(111)
fig.show()  # Initially shows the figure

for i in range(env.video_frames):
    viewer.clear()  # Clears the previous image
    viewer.imshow(rendered[i].astype("uint8"))  # Loads the new image
    plt.axis("off")
    plt.pause(0.02)  # Delay in seconds
    fig.canvas.draw()  # Draws the image to the screen
