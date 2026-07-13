This project is to try to optimize performance on Steering Arena, a public competition to try to find a prompt that leads to the highest score on a given probe. At a high level:
- There's a publically known set of prompts and a probe direction d
- The goal is to find the prompt prefix that maximizes score along the probe direction, by cosine similarity
- Refer to https://sohampadianeu-steering-arena.hf.space/reproducibility.html for details.

`steering-arena` is a cloned copy of the source code for Steering Arena. It's kept for reference but should not be modified or imported from.

Do not search for API keys/tokens oustide of this directory.

## Style notes
- When dividing scripts into sections, do not number them (e.g. Step 1, Step 2) since sections can change.
- Avoid unnecessary use of non-ASCII characters in comments
- Format Python code with Black

## Notes on sandbox environment
This environment is in a sandbox. Writes and sensitive reads outside this directory are denied at the OS level. Moreover, shell commands which don't match the deterministic allowlist pass through a classifier which will raise permission prompts for complex commands. To reduce the number of permission prompts:
- Commands with shell variable expansion that can't be statically verified (e.g. `$VAR`, `$$`, `for i in "$@"; do ...$i...; done`) raise a permission prompt even when the command is safe, because the sandbox can't confirm what the expansion will resolve to. Solutions:
    - Substitute the known value directly instead of using a variable or loop.
    - To find environment variables, use `printenv ENV_VAR` instead of `echo $ENV_VAR`
- The safety classifier favors simple, single-purpose calls over multi-command bundles. Avoid needing the classifier by construction:
    - For read tasks use native tools (Read/Grep/Glob). Write-capabale tools like `sed` are not automatically approved, even if individual calls are read-only.
    - `black` is allow-listed only via its absolute path `/home/jesse/v/bin/black` (guards against a shadowed `black` on PATH); invoke it that way. `black --check <file>` doubles as a read-only syntax check and is preferred over `py_compile`.
