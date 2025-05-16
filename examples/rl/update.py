import gym
from deep_q_network import DeepQNetwork
import numpy as np


# Initialize DQN
def update():
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    # Load checkpoint
    model_path = "weights/CartPole-v0.ckpt"

    DQN = DeepQNetwork(  n_y=env.action_space.n,
                        n_x=env.observation_space.shape[0],
                        learning_rate=0.01,
                        replace_target_iter=100,
                        memory_size=2000,
                        batch_size=32,
                        epsilon_max=0.9,
                        epsilon_greedy_increment=0.001,
                        load_path=model_path

                    )

    DQN.learn()


if __name__ == "__main__":
    update()