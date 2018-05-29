## World Models

Implementation of [World Models](https://arxiv.org/pdf/1803.10122.pdf) for the Crab2D environment.

The main idea of World Models is to separately train V, M, and C modules.  The challenge for locomotion tasks is that a random controller can only explore very limited portion of the state space.  In order to construct accurate models for V and M, the training process should be iterative.

Some key points:
* The original paper is about a generative model for building approximate models of popular RL environments, which is why they used a VAE.  Simple autoencoder should work for control tasks.
* The use of MDN-RNN is used to model random environments, though they did find benefits in randomness for transferring to real simulators.  Perhaps using M to directly predict next hidden state is good enough, rather than combining it with mixture density network.   