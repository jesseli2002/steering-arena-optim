This project is to try to optimize performance on Steering Arena, a public competition to try to find a prompt that leads to the highest score on a given probe.

`steering-arena` is a cloned copy of the source code for Steering Arena and is kept here for reference, but should not be modified.

Do not search for API keys/tokens oustide of this directory.

## Notes on sandbox environment
- This environment runs commands through a sandbox that permission-checks each Bash
invocation. Commands that build shell expansions dynamically inside a loop (e.g.
`for i in "$@"; do ...$i...; done`) can't be statically verified, so the sandbox
falls back to a permission prompt even when the command is safe. Where practical,
prefer writing out the expanded values directly instead of looping, to avoid
unnecessary prompts.
