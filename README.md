# Activation Steering with Greedy Coordinate Gradients

I stumbled upon someone's competition ([Steering Arena](https://sohampadianeu-steering-arena.hf.space/)) to find a prompt for a LLM that maximizes how close the model's activation are to a given probe direction. When I found it, all of the submissions were probably human-written, and in actual English. I looked at this, and figured "Hey - this would be a good first project to practice the empirical AI safety things I've been learning!"

One way to tackle this is with Greedy Coordinate Gradients, which I found about in [Accelerating Greedy Coordinate Gradient and General Prompt Optimization via Probe Sampling](https://arxiv.org/abs/2307.15043). That paper optimizes a prompt in an attempt to find a universal jailbreak for LLMs, with a substantial amount of effort and insight being what the objective function even should be. Luckily for me, this project doesn't face that issue; my objective is simply the cosine similarity between the activations at a particular layer and a specified probe direction.

Slightly more formally, the problem is as follows: We control a prefix string (`prefix`), which gets prepended to one of several uncontrolled but known prompts (`suffix`).
The most immediate problem is that optimizing over tokens is a discrete problem, whereas optimizers tend to prefer continuous spaces. However, we basically immediately embed the tokens into a continuous space (one-hot embedding), so the simple workaround is to optimize in the one-hot embedding space.
At a high level, each iteration of the optimization algorithm works like this (Algorithm 1):
- Run a forward pass on your data, and evaluate the gradient of the score with respect to the one-hot embeddings.
- For each token position, find the tokens (which correspond to the vocab dimension) with the highest gradients. Pick the top `k` such tokens.
- On each iteration, generate `B` candidates (the "batch size").
    - Each candidate modifies a single token from the current `prefix`. Which token is modified is randomly selected (uniformly over token positions), and what it's modified to is also randomly selected (uniformly over the top `k` tokens we found in the previous step)
- Across your `B` candidates, pick the best performing one as your new `prefix`.

### Issues
Some of the questions I had, and partial answers:

1. What happens if the resulting prompt has a different tokenization than the one we optimized for?
    - It kind of still works... Models seem to be at least somewhat robust to non-canonical tokenizations, although in the limit you'll see performance degradation
    - Probably what you could do, is once you near convergence, run a tokenization pass, then re-initialize your optimizer with your new tokens. If you get more tokens as a result, try prepending the prefix.
    - There's probably inevitably going to be some weird behaviour at the end, where you join the prefix to the suffix. I don't know if there's a smart way to deal with this, but also it maybe doesn't matter that much.
2. Why sample uniformly over the top-k, rather than some smarter way that gives higher weight to higher gradients?
    - It's easier, for one
    - There's probably improvements in this direction. The optimization literature has been around for a long time; I wouldn't be surprised if there was some standard formula based on entropy or thermodynamics or something that gives optimal performance for some definition of optimal

For basically all of these, I plan on trying some things out - stay tuned!



