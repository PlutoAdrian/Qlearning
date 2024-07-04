import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from IPython.display import clear_output
import random


def save_frames_as_gif(frames, episode, algorithm_type, path='./gif/', filename='gym_animation.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(os.path.join(path, filename), writer='imagemagick', fps=10)

env = gym.make('CliffWalking-v0', render_mode="rgb_array")

ALPHA = 0.75
EPSILONMIN = 0.02
EPSILON = 1.00
EPSILONRATE = 0.99
RENDER_AT_EPISODE = 20
EPISODES = 200
LEARNING_RATE = 0.75
alpha = np.linspace(0.01, 0.99, 10)
learning_rate = np.linspace(0.01, 0.99, 10)
episode_frames_Q = {}
time_step = 0
sum_of_rewards = []

Q_table = np.zeros((env.observation_space.n, env.action_space.n))  
for episode in range(EPISODES): 
    actionc = 0  
    observation, info = env.reset()  
    done = False
    rewards = 0
    frames = []
    while ((not done)&(actionc < 1000)): 
        if random.random() < max(EPSILONMIN, ((EPISODES - episode) / EPISODES)):
        #if random.random() < max(EPSILONMIN, EPSILON):
            action = random.randint(0, env.action_space.n - 1)
        else: 
            action = np.argmax(Q_table[observation])

        actionc += 1
        
        next_observation, reward, done, truncated, info = env.step(action)
        Q_table[observation][action] = Q_table[observation][action] + ALPHA * (reward + LEARNING_RATE * np.max(Q_table[next_observation]) - Q_table[observation][action])
        observation = next_observation
        time_step += 1
        rewards += reward
        if episode % RENDER_AT_EPISODE == 0:
            clear_output(wait=True)
            single_frame = env.render()
            frames.append(single_frame)
    sum_of_rewards.append(rewards)
    EPSILON = EPSILON * EPSILONRATE
    if episode % RENDER_AT_EPISODE == 0:
        frames.append(single_frame)
        frames.append(single_frame)
        episode_frames_Q[episode] = frames
        save_frames_as_gif(frames, episode, "Q-Learning", filename=f"Q-Learning episode {episode}.gif")
plt.figure(figsize=(10, 5))
plt.plot(range(EPISODES), sum_of_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode')
plt.grid(True)
plt.savefig('./graph/')