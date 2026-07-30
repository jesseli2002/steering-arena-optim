# [Activation Steering with Greedy Coordinate Gradients](https://jesseli2002.github.io/blog/projects/gcg-activation-steering/)

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

## Code
There's not any complicated structure here:
- steering-arena is included as a git submodule, to get access to the underlying data (probe directions & test prompts)
- optimize_prompt.py is the actual optimization script
    - Set the environment variable HF_TOKEN (or put it in a .env file) to be a personal access token for HuggingFace, so that the Olmo model weights can get downloaded.
- print_best.py is a simple helper to print the best or latest prompt; I use it and pipe it into `xclip` so I can copy and paste it into the Steering Arena website
- results/ holds selected training results (by default, the optimizer outputs results to `data/`, a gitignore'd directory)
- results.py makes the plots [in my writeup](https://jesseli2002.github.io/blog/projects/gcg-activation-steering/), putting them under plot/
