import random
from dataclasses import dataclass
from abc import ABC, abstractmethod, abstractproperty
from typing import Any

import torch
import numpy as np
import numpy.typing as npt


class Action(ABC):
    @abstractmethod
    def gymnasium(self) -> Any:
        ...

    @abstractmethod
    def tensor(self) -> torch.Tensor:
        ...


@dataclass
class DiscreteAction(Action):
    action: int

    def gymnasium(self) -> int:
        return self.action

    def tensor(self) -> torch.Tensor:
        return torch.tensor([self.action])


@dataclass
class ContinuousAction(Action):
    action: torch.Tensor  # of float(s)

    def gymnasium(self) -> np.ndarray:
        return self.action.detach().cpu().numpy()

    def tensor(self) -> torch.Tensor:
        return self.action

    def __add__(self, other) -> "ContinuousAction":
        return ContinuousAction(self.action + other)


@dataclass
class State:
    tensor: torch.Tensor  # 1D array
    terminal: bool


@dataclass
class Transition:
    """The result of taking A_t in S_t, obtaining R_t and transitionning
    to S_t+1."""

    action: Action  # A_t
    old_state: State  # S_t
    new_state: State  # S_t+1
    reward: float  # R_t
    truncated: bool  # out of timesteps

    def end_of_episode(self) -> bool:
        return self.new_state.terminal or self.truncated


class Environment(ABC):
    @abstractmethod
    def won(self, transition: Transition) -> bool:
        ...

    @abstractproperty
    def observation_space_length(self) -> int:
        ...

    @abstractproperty
    def action_count(self) -> int:
        ...

    @abstractmethod
    def random_action(self) -> Action:
        ...

    @abstractmethod
    def take_action(self, action: Action) -> Transition:
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractproperty
    def current_state(self) -> State:
        ...

    @abstractproperty
    def needs_reset(self) -> bool:
        ...

    @abstractproperty
    def last_reward(self) -> float:
        ...


class DiscreteActionEnv(Environment):
    @abstractproperty
    def action_list(self) -> list[DiscreteAction]:
        ...

    @property
    def action_count(self) -> int:
        return len(self.action_list)

    def random_action(self) -> DiscreteAction:
        return random.choice(self.action_list)


class ContinuousActionEnv(Environment):
    @property
    def action_count(self) -> int:
        ...
