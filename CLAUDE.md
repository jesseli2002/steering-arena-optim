This project is to try to optimize performance on Steering Arena, a public competition to try to find a prompt that leads to the highest score on a given probe. At a high level:
- There's a publically known set of prompts and a probe direction d
- The goal is to find the prompt prefix that maximizes score along the probe direction, by cosine similarity
- Refer to https://sohampadianeu-steering-arena.hf.space/reproducibility.html for details.
- The git remote is GitLab, not GitHub.

`steering-arena` is a cloned copy of the source code for Steering Arena. It's kept for reference but should not be modified or imported from.

Do not search for API keys/tokens oustide of this directory.

## Style notes
- When dividing scripts into sections, do not number them (e.g. Step 1, Step 2) since sections can change.
- Avoid unnecessary use of non-ASCII characters in comments
- Format Python code with Black
- torch tensors on GPU should be indexed with CPU - not only is this allowed, it's less brittle, since if the indexing tensor is on a different GPU than the indexed tensor, the operation fails.


## vast_setup
`vast_setup/` is a separate nested git repo. To modify its files, `/cd` into it first — editing it from an outer-rooted session hits worktree-isolation friction. If asked to change vast_setup while rooted at the outer repo, remind the user to `/cd vast_setup` before starting.

## Background processes
Each Bash call runs in its own PID namespace, so a sandboxed `ps` sees only its own invocation, and plain `&`/`nohup`/`setsid` processes are killed when the invocation ends.
- Never use `ps` to check whether a process you launched is still alive; empty output means "not visible from here," not "dead."
- For long-running work, launch it with `run_in_background`, monitor via its output file and completion notification, and stop it with the TaskStop tool (which reaps the whole subprocess tree via the PID namespace).
- To inspect or kill a host process, or anything the harness isn't tracking, hand the user a host `ps`/`kill` command rather than disabling the sandbox.
- Above all, be cautious - there's still quirks in the sandbox environment setup. If a process (especially a resource-intensive one) should be running but you can't find it, don't assume it died unexpectedly; check if your harnesses can tell the difference between a dead and invisible process, and don't be afraid to ask the user for help diagnosing issues.

## Notes on sandbox environment
This environment is in a sandbox. Writes and sensitive reads outside this directory are denied at the OS level. Moreover, shell commands which don't match the deterministic allowlist pass through a classifier which will raise permission prompts for complex commands. To reduce the number of permission prompts:
- Commands with shell variable expansion that can't be statically verified (e.g. `$VAR`, `$$`, `for i in "$@"; do ...$i...; done`) raise a permission prompt even when the command is safe, because the sandbox can't confirm what the expansion will resolve to. Solutions:
    - Substitute the known value directly instead of using a variable or loop.
    - To find environment variables, use `printenv ENV_VAR` instead of `echo $ENV_VAR`
- The safety classifier favors simple, single-purpose calls over multi-command bundles. Avoid needing the classifier by construction:
    - For read tasks use native tools (Read/Grep/Glob). Write-capabale tools like `sed` are not automatically approved, even if individual calls are read-only.
    - `black` is allow-listed only via its absolute path `/home/jesse/v/bin/black` (guards against a shadowed `black` on PATH); invoke it that way. `black --check <file>` doubles as a read-only syntax check and is preferred over `py_compile`.

### vast_setup
`vast_setup/` is a separate nested git repo. To modify its files, `/cd` into it first — editing it from an outer-rooted session hits worktree-isolation friction. IF WORKING ON FILES IN `vast_setup/`, REFUSE WORK UNTIL USER HAS `/cd`'D INTO `vast_setup/`.
