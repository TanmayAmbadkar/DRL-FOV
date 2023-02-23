from data_loader.data_loader import load_data
from environment.environment import Environment
from agent.agent import Agent
from train.train import train
import torch
import matplotlib.pyplot as plt
import numpy as np

images_dir = "360_saliency_dataset_2018eccv/output"
images_color, images_saliency = load_data(images_dir)

# TRAINING

env = Environment(saliency_frames = images_saliency, colored_frames = images_color, n_frames = 2, video_frames=200, frame_width=160, frame_height=90)

agent = Agent(n_frames = 2)

agent = train(agent, env, episodes=100)

torch.save(agent.model.state_dict(), "saved_agents/agent2frame.model")
# Testing

rendered = np.zeros((env.video_frames,env.frame_height, env.frame_width, 3))
observation = env.reset()
batch_size = 12
for episode in range(1):
    done = False
    count = 0
    reward_episode = []
    while not done:
        rendered[count:count+env.n_frames] = env.render()
        count+=env.n_frames
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
fig.show() # Initially shows the figure

for i in range(env.video_frames):
    viewer.clear() # Clears the previous image
    viewer.imshow(rendered[i].astype('uint8')) # Loads the new image
    plt.axis('off')
    plt.pause(.02) # Delay in seconds
    fig.canvas.draw() # Draws the image to the screen
