"""
Basic tests for the QLearningAgent to ensure correct initialisation
and action selection behaviour.
"""

from src.reinforcement.q_learning import QLearningAgent


def test_qtable_shape():
    agent = QLearningAgent(rows=5, cols=7)
    assert agent.q_table.shape == (5, 7, 4)


def test_choose_action_in_range():
    agent = QLearningAgent(rows=3, cols=3)
    state = (0, 0)
    action = agent.choose_action(state)
    assert 0 <= action <= 3