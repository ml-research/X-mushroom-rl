import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from x_mushroom_rl.algorithms.value import DQN
from x_mushroom_rl.approximators.parametric import TorchApproximator
from x_mushroom_rl.core import Core
from x_mushroom_rl.environments import Atari
from x_mushroom_rl.policy import EpsGreedy
from x_mushroom_rl.utils.dataset import compute_metrics
from x_mushroom_rl.utils.parameters import LinearParameter, Parameter
from x_mushroom_rl.features.image_extractors.interactive_color_extractor import IColorExtractor as ColorExtractor
# from x_mushroom_rl.features.image_extractors.color_extractor import ColorExtractor


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def get_stats(dataset):
    score = compute_metrics(dataset)
    print(('min_reward: %f, max_reward: %f, mean_reward: %f,'
          ' games_completed: %d' % score))

    return score


class LinearClassifier(nn.Module):
    n_features = 512

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        self._h5 = nn.Linear(n_input, n_output)

    def forward(self, state, action=None):
        q = self._h5(state.float())
        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))
            return q_acted



scores = list()

optimizer = dict()
optimizer['class'] = optim.Adam
optimizer['params'] = dict(lr=.00025)

# Settings
width, height = None, None
history_length = 4
train_frequency = 4
evaluation_frequency = 2500
target_update_frequency = 100
initial_replay_size = 500
max_replay_size = 5000
test_samples = 1250
max_steps = 50000000

# MDP
game = ["Carnival", "MsPacman", "Pong", "SpaceInvaders", "Tennis"][0]
mdp = Atari(f'{game}Deterministic-v4', width, height, ends_at_life=True,
            history_length=history_length, max_no_op_actions=30)

# Policy
epsilon = LinearParameter(value=1.,
                          threshold_value=.1,
                          n=1000000)
epsilon_test = Parameter(value=.05)
epsilon_random = Parameter(value=1)
pi = EpsGreedy(epsilon=epsilon_random)

feature_extractor = ColorExtractor(game=game)
# feature_extractor.show_objects = True

# Approximator
input_shape = (history_length*feature_extractor.max_obj*2,)
approximator_params = dict(
    network=LinearClassifier,
    input_shape=input_shape,
    output_shape=(mdp.info.action_space.n,),
    n_actions=mdp.info.action_space.n,
    n_features=512,
    optimizer=optimizer,
    loss=F.smooth_l1_loss
)

approximator = TorchApproximator

# Agent
algorithm_params = dict(
    batch_size=32,
    target_update_frequency=target_update_frequency // train_frequency,
    replay_memory=None,
    initial_replay_size=initial_replay_size,
    max_replay_size=max_replay_size
)

agent = DQN(mdp.info, pi, approximator,
            approximator_params=approximator_params,
            **algorithm_params)


agent.phi = feature_extractor
# Algorithm
core = Core(agent, mdp)

# RUN

# Fill replay memory with random dataset
print_epoch(0)
core.learn(n_steps=initial_replay_size,
           n_steps_per_fit=initial_replay_size)

# Evaluate initial policy
pi.set_epsilon(epsilon_test)
mdp.set_episode_end(False)
dataset = core.evaluate(n_steps=test_samples)
scores.append(get_stats(dataset))

for n_epoch in range(1, max_steps // evaluation_frequency + 1):
    print_epoch(n_epoch)
    print('- Learning:')
    # learning step
    pi.set_epsilon(epsilon)
    mdp.set_episode_end(True)
    core.learn(n_steps=evaluation_frequency,
               n_steps_per_fit=train_frequency)

    print('- Evaluation:')
    # evaluation step
    pi.set_epsilon(epsilon_test)
    mdp.set_episode_end(False)
    dataset = core.evaluate(n_steps=test_samples, render=True)
    scores.append(get_stats(dataset))
