import numpy as np
from src.reinforcement.q_learning import QLearningAgent
from src.reinforcement.gym_env import GridEnv


def train_gym_qlearning(
    rows,
    cols,
    obstacles,
    start,
    goal,
    episodes=500,
    max_steps=500
):
    """
    Train Q-learning using the Gym environment.
    """

    env = GridEnv(rows, cols, obstacles, start, goal)
    agent = QLearningAgent(rows, cols)

    episode_rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            state = tuple(obs.tolist())      # convert np.array -> (r, c)

            action = agent.choose_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)

            next_state = tuple(obs.tolist())

            agent.update(state, action, reward, next_state)

            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        agent.decay_epsilon()

    return agent, episode_rewards