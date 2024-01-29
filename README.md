# Trackmania DQN

A modular implementation of both the [DQN](https://en.wikipedia.org/wiki/Q-learning#Deep_Q-learning) and [DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) reinforcement learning algorithms, trained on the [Trackmania](https://en.wikipedia.org/wiki/Trackmania_(2020_video_game)) (2020) videogame.

DQN has a discrete action space, and as such uses keyboard input i.e. can go left, right or straight (no in-between).   
DDPG on the other hand uses analog input, and can control how sharply it turns.

The algorithms are implemented modularly such that they can be used with any environment, including those from [OpenAI Gymnasium](https://gymnasium.farama.org/index.html). They were battle-tested on Cartpole, Pendulum, MountainCar (discrete + continuous) and Lunar Lander (discrete + continuous) before Trackmania.

DQN is implemented with ε-decay, Polyak updates, [Prioritised Experience Replay](https://doi.org/10.48550/arXiv.1511.05952) and [Double-DQN](https://doi.org/10.1609/aaai.v30i1.10295).

Trackmania integration was