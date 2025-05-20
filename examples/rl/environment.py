"""
Deep Q Network
CartPole-v0
"""

import gym
from deep_q_network import DeepQNetwork
import numpy as np
def run():
    env = gym.make('CartPole-v0')
    env = env.unwrapped

    model_path = "weights/CartPole-v0.ckpt"

    DQN = DeepQNetwork(  n_y=env.action_space.n,
                        n_x=env.observation_space.shape[0],
                        learning_rate=0.01,
                        replace_target_iter=100,
                        memory_size=2000,
                        batch_size=32,
                        epsilon_max=0.9,
                        epsilon_greedy_increment=0.001,
                        model_path=model_path

                    )


    RENDER_ENV = False
    rewards = []
    RENDER_REWARD_MIN = 800
    total_steps_counter = 0

    for episode in range(400):

        observation = env.reset()
        episode_reward = 0

        while True:
            if RENDER_ENV: env.render()

            action = DQN.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
            reward = r1 + r2

            DQN.store_transition(observation, action, reward, observation_)

            episode_reward += reward

            if done:
                rewards.append(episode_reward)
                max_reward_so_far = np.amax(rewards)

                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", round(episode_reward, 2))
                print("Epsilon: ", round(DQN.epsilon,2))
                print("Max reward so far: ", max_reward_so_far)

                if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True

                break

            observation = observation_

            total_steps_counter += 1

if __name__ == "__main__":
    run()