Repo: /Users/mike/Projects/KovaForge/gbrain
Task: Sync from upstream garrytan/gbrain to fork KovaForge/gbrain, rebuild gbrain CLI with bun, install binary to ~/.local/bin/gbrain, confirm with gbrain --version.
Target: gbrain CLI (built via bun)
Constraints:
- Binary must end up at ~/.local/bin/gbrain as a standalone file (NOT a symlink)
- Do not disrupt any existing storage or data
- Always remove the existing ~/.local/bin/gbrain BEFORE copying the new binary (prevents symlink-follow overwriting source files)

Steps:
1. In /Users/mike/Projects/KovaForge/gbrain, fetch from upstream: `git fetch upstream`
2. Check for new commits: `git log --oneline upstream/master ^HEAD` (if empty, already up to date)
3. Merge upstream/master into your current branch: `git merge upstream/master`
   - If conflicts: resolve them. Prefer upstream version for test files, keep KovaForge local changes for src/cli.ts, package.json, and bun.lock
   - If local modifications exist: stash them first with `git stash`, do the merge, then `git stash pop` to reapply
4. Build the gbrain CLI using the project's build process: `bun run build`
   - From package.json this is: `bun build --compile --outfile bin/gbrain src/cli.ts`
5. CRITICAL: Remove existing binary symlink/file before installing:
   - `rm -f ~/.local/bin/gbrain` (removes symlink or file)
6. Copy the new binary: `cp bin/gbrain ~/.local/bin/gbrain && chmod +x ~/.local/bin/gbrain`
7. Confirm: `gbrain --version`
8. Report: version, git merge summary (commits merged, conflicts resolved), and any stash/applying notes.