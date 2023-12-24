import os
from dataclasses import dataclass
from typing import cast, TYPE_CHECKING, Any, Iterable

import torch
from torch import nn
from torch.nn.functional import relu


# prevent circular import
if TYPE_CHECKING:
    from environment import State, Environment, Action
    from dqn import ExperienceBatch
else:
    Experience = object
    State = object
    Action = object
    Environment = object
    ExperienceBatch = object


@dataclass
class NeuralNetworkResult:
    tensor: torch.Tensor

    def best_action(self) -> Action:
        argmax: torch.Tensor = self.tensor.argmax()  # this is a tensor with one item
        best_action = argmax.item()
        return cast(Action, best_action)

    def best_action_q_value(self) -> float:
        return self.tensor[self.best_action()].item()

    def q_value_for_action(self, action: Action) -> float:
        return self.tensor[action].item()


class NeuralNetworkResultBatch:
    def __init__(self, batch_output: torch.Tensor):
        # Tensor[[QValue * 3], [QValue * 3], ...]
        self.batch_output = batch_output

    def __getitem__(self, index: int) -> NeuralNetworkResult:
        """Override index operator e.g. batch[0] -> NeuralNetworkResult"""
        return NeuralNetworkResult(self.batch_output[index])


class NeuralNetwork(nn.Module):
    @staticmethod
    def device() -> torch.device:
        """Utility function to determine whether we can run on GPU"""
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        return torch.device(device)

    @staticmethod
    def tensorify(array: Iterable) -> torch.Tensor:
        """Create a PyTorch tensor, and make sure it's on the GPU if possible"""
        return torch.tensor(array, device=NeuralNetwork.device())

    def __init__(self, env: Environment):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(env.observation_space_length, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_count),
        )

        # self.optim = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        self.optim = torch.optim.AdamW(self.parameters(), lr=1e-2, amsgrad=True)

    # do not call directly, call get_q_values() instead
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """PyTorch internal function to perform forward pass.
        Do not call directly, use get_q_values() instead.

        Args:
            state (torch.Tensor): a tensor of length 6 (the state has 6 variables)

        Returns:
            torch.Tensor: a tensor of length 3 (one q-value for each action)
        """

        return self.linear_relu_stack(state)

    # need to return the q value for an action AND
    # return the corresponding action so DQN class knows what to use
    def get_q_values(self, state: State) -> NeuralNetworkResult:
        """For a given state, pass it through the neural network and return
        the q-values for each action in that state.

        Args:
            state (State): The state to get the q-values for.

        Returns:
            NeuralNetworkResult: An object that wraps the raw tensor, with
                utility methods such as q_value_for_action() to make our lives
                easier.
        """

        neural_network_output = self(state)
        return NeuralNetworkResult(neural_network_output)

    def get_q_values_batch(self, states: torch.Tensor) -> NeuralNetworkResultBatch:
        # states = Tensor[State, State, ...]
        # where State is Tensor[position, velocity]

        batch_output = self(states)
        # batch_output = Tensor[[QValue * 3], [QValue * 3], ...]
        # where QValue is float

        return NeuralNetworkResultBatch(batch_output)

    def get_best_action(self, state: State) -> Action:
        """Get the best action in a given state according to the neural network.

        Args:
            state (State): The state to get the best action for.

        Returns:
            Action: The best action in the given state.
        """

        neural_network_result = self.get_q_values(state)
        return neural_network_result.best_action()

    def backprop(
        self, experiences: ExperienceBatch, policy_net_qvalues: NeuralNetworkResultBatch
    ):
        self.optim.zero_grad()

        # Tensor[State, State, ...]
        # where State is Tensor[position, velocity]
        experience_states = experiences.old_states

        # Tensor[[QValue * 3], [QValue * 3], ...]
        # where QValue is float
        q_values = self(experience_states)

        # Tensor[[Action], [Action], ...]
        # where Action is int
        actions_chosen = experiences.actions

        # Tensor[[QValue], [QValue], ...]
        actions_chosen_q_values = q_values.gather(1, actions_chosen)

        # Tensor[[TDTarget], [TDTarget], ...]
        # where TDTarget is QValue
        td_targets_tensor = policy_net_qvalues.batch_output.gather(1, actions_chosen)

        criterion = torch.nn.MSELoss()
        loss = criterion(actions_chosen_q_values, td_targets_tensor)
        loss.backward()

        self.optim.step()  # gradient descent
