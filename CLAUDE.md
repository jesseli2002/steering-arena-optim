This project is to try to optimize performance on Steering Arena, a public competition to try to find a prompt that leads to the highest score on a given probe.

`steering-arena` is a cloned copy of the source code for Steering Arena. It's kept for reference but should not be modified.

Do not search for API keys/tokens oustide of this directory.

## Style notes
- When dividing scripts into sections, do not number them (e.g. Step 1, Step 2) since sections can change.
- Avoid unnecessary use of non-ASCII characters in comments

## Notes on sandbox environment
- This environment runs commands through a sandbox that permission-checks each Bash invocation. Any command containing shell variable expansion that can't be statically verified (e.g. `$VAR`, `$$`, `for i in "$@"; do ...$i...; done`) falls back to a permission prompt even when the command is safe, because the sandbox can't confirm what the expansion will resolve to. Solutions:
    - Substitute the literal/known value directly instead of using a variable or loop, to avoid unnecessary prompts.
    - To find environment variables, use `printenv ENV_VAR` instead of `echo $ENV_VAR`
