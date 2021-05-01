# PongRL
A quick project designed to explore reinforcement learning. Uses a neural network to approximate the state-action function, and then updates the weights using gradient Q-learning. This is highly unstable, but seems to work well if given the right rewards.

One note is that there is a pretty high chance of divergence if it doesn't seem to converge to a perfect strategy within about a minute. My laptop struggles to handle the updates, so I've never let it run for too much longer than that to see if it will eventually converge but my guess is that it never gets out of its diverging policy.
