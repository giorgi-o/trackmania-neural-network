import random
from dataclasses import dataclass
import math
import collections
from typing import Iterable

import torch
import numpy as np

from environment import Environment, Action, State, Transition
from network import NeuralNetwork, NeuralNetworkResult
from data_helper import plot_episode_data, EpisodeData


@dataclass
class TdTargetBatch:
    # Tensor[TDTarget, TDTarget, ...]
    tensor: torch.Tensor


@dataclass
class Experience:
    transition: Transition
    td_error: float


class ExperienceBatch:
    def __init__(self, replay_buffer: "ReplayBuffer", experiences: list[Experience]):
        self.replay_buffer = replay_buffer
        self.experiences = experiences
        self.size = len(experiences)

        # Tensor[[0], [2], [1], ...]
        self.actions = NeuralNetwork.tensorify([[exp.transition.action] for exp in experiences])

        # Tensor[-0.99, -0.99, ...]
        self.rewards = NeuralNetwork.tensorify([exp.transition.reward for exp in experiences])

        # Tensor[State, State, ...]
        # states are already torch tensors, so we can just use torch.stack
        self.old_states = torch.stack([exp.transition.old_state.tensor for exp in experiences])
        self.new_states = torch.stack([exp.transition.new_state.tensor for exp in experiences])

        # Tensor[False, False, True, ...]
        self.terminal = NeuralNetwork.tensorify([exp.transition.new_state.terminal for exp in experiences])

    def update_td_errors(self, td_errors: Iterable[float]):
        for exp, td_error in zip(self.experiences, td_errors):
            exp.td_error = td_error


class ReplayBuffer:
    def __init__(self, max_len=10000, omega=0.5):
        self.buffer = collections.deque(maxlen=max_len)
        self.omega = omega

    def add_experience(self, transition: Transition):
        experience = Experience(transition, float("inf"))
        self.buffer.append(experience)

    def get_batch(self, batch_size: int) -> ExperienceBatch:
        priorities = [abs(exp.td_error) ** self.omega for exp in self.buffer]
        priorities = np.array(priorities)
        priorities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=priorities)
        experiences = [self.buffer[idx].transition for idx in indices]

        return ExperienceBatch(self, experiences)

    def size(self) -> int:
        return len(self.buffer)


class DQN:
    def __init__(
        self,
        episode_count: int,
        timestep_count: int,
        gamma: float,
        epsilon_start: float = 0.9,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.05,
        C: int = 50,
        buffer_batch_size: int = 100,
    ):
        self.episode_count = episode_count
        self.timestep_count = timestep_count

        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_rate = epsilon_decay
        self.epsilon = epsilon_start

        self.gamma = gamma
        self.C = C
        self.buffer_batch_size = buffer_batch_size

        self.environment = Environment()

        # initialise replay memory
        self.replay_buffer = ReplayBuffer()
        # initialise q1
        self.policy_network = NeuralNetwork(self.environment).to(NeuralNetwork.device())
        # initialise q2
        self.target_network = NeuralNetwork(self.environment).to(NeuralNetwork.device())
        # copy q2 to q1
        self.policy_network.load_state_dict(self.target_network.state_dict())

    def get_best_action(self, state: State) -> Action:
        return self.policy_network.get_best_action(state)

    def get_action_using_epsilon_greedy(self, state: State):
        if np.random.uniform(0, 1) < self.epsilon:
            # pick random action
            action = random.choice(self.environment.action_list)
        else:
            # pick best action
            action = self.get_best_action(state)
        return action

    def execute_action(self, action: Action) -> Transition:
        return self.environment.take_action(action)

    # using policy
    def get_q_values(self, state: State) -> NeuralNetworkResult:
        return self.policy_network.get_q_values(state)

    # using target network here to estimate q values
    def get_q_value_for_action(self, state: State, action: Action, policy_net=False) -> float:
        network = self.policy_network if policy_net else self.target_network
        neural_network_result = network.get_q_values(state)
        return neural_network_result.q_value_for_action(action)

    def compute_td_target(self, experience: Transition) -> float:
        # TD Target is the last reward + the expected reward of the
        # best action in the next state, discounted.

        # the reward and state after the last action was taken:
        last_reward = experience.reward  # R_t
        current_state = experience.new_state  # S_t+1

        if current_state.terminal:  # terminal experience
            td_target = last_reward
        else:
            target_net_result = self.target_network.get_q_values(current_state)
            best_q_value = target_net_result.best_action_q_value()
            td_target = last_reward + self.gamma * best_q_value

        return td_target

    def compute_td_targets_batch(self, experiences: ExperienceBatch) -> TdTargetBatch:
        # td target is:
        # reward + discounted qvalue  (if not terminal)
        # reward + 0                  (if terminal)

        # Tensor[-0.99, -0.99, ...]
        rewards = experiences.rewards

        # Tensor[[QValue * 3], [QValue * 3], ...]
        discounted_qvalues = self.target_network.get_q_values_batch(experiences.new_states)
        discounted_qvalues_tensor = discounted_qvalues.tensor

        # pick the QValue associated with the best action
        # Tensor[QValue, QValue, ...]
        discounted_qvalues_tensor = discounted_qvalues_tensor.max(1).values
        discounted_qvalues_tensor[experiences.terminal] = 0
        discounted_qvalues_tensor *= self.gamma

        # Tensor[TDTarget, TDTarget, ...]
        td_targets = rewards + discounted_qvalues_tensor
        return TdTargetBatch(td_targets)

    def update_experiences_td_errors(self, experiences: ExperienceBatch):
        td_targets = self.compute_td_targets_batch(experiences)

        q_values = self.target_network.get_q_values_batch(experiences.old_states)
        q_values = q_values.for_actions(experiences.actions)

        td_errors = (td_targets - q_values) ** self.replay_buffer.omega
        td_errors = td_errors.numpy()
        experiences.update_td_errors(td_errors)

    def decay_epsilon(self, episode):
        # epsilon = epsilon_min + (epsilon_start - epsilon_min) x epsilon^-decay_rate * episode
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * math.exp(
            -self.decay_rate * episode
        )

    def update_target_network(self):
        target_net_state = self.target_network.state_dict()
        policy_net_state = self.policy_network.state_dict()
        tau = 0.005

        for key in policy_net_state:
            target_net_state[key] = tau * policy_net_state[key] + (1 - tau) * target_net_state[key]

        self.target_network.load_state_dict(target_net_state)

    def backprop(self, experiences: ExperienceBatch, td_targets: TdTargetBatch):
        self.policy_network.backprop(experiences, td_targets)

    def train(self):
        episodes = []

        try:
            timestep_C_count = 0
            recent_rewards = collections.deque(maxlen=30)
            for episode in range(self.episode_count):
                self.environment.reset()

                reward_sum = 0
                transition = None
                timestep = 0

                for timestep in range(self.timestep_count):
                    state = self.environment.current_state  # S_t
                    action = self.get_action_using_epsilon_greedy(state)  # A_t

                    transition = self.execute_action(action)
                    reward_sum += transition.reward
                    self.replay_buffer.add_experience(transition)

                    # print(
                    #     f"Episode {episode} Timestep {timestep} | Action {action}, Reward {action_result.reward:.0f}, Total Reward {reward_sum:.0f}"
                    # )

                    if self.replay_buffer.size() > self.buffer_batch_size:
                        replay_batch = self.replay_buffer.get_batch(self.buffer_batch_size)
                        td_targets = self.compute_td_targets_batch(replay_batch)

                        self.backprop(replay_batch, td_targets)
                        self.update_experiences_td_errors(replay_batch)

                    timestep_C_count += 1
                    if timestep_C_count == self.C:
                        self.update_target_network()
                        timestep_C_count = 0

                    # process termination
                    if transition.end_of_episode():
                        break

                # episode ended
                recent_rewards.append(reward_sum)

                # print episode result
                assert transition is not None
                won = transition.truncated
                won_str = "(won) " if won else "(lost)"
                running_avg = sum(recent_rewards) / len(recent_rewards)
                print(
                    f"Episode {episode+1: <3} | {timestep+1: >3} timesteps {won_str}"
                    f" | reward {reward_sum: <6.2f} | avg {running_avg: <6.2f} (last {len(recent_rewards): <2})"
                    f" | ε {self.epsilon:.2f}"
                )

                episodes.append(EpisodeData(episode, reward_sum, timestep, won))
                self.decay_epsilon(episode)
                # print(f"Episode {episode} finished with total reward {reward_sum}")

        except KeyboardInterrupt:
            pass

        try:
            plot_episode_data(episodes)
        except KeyboardInterrupt:
            pass  # ctrl-c to close plot
