# CLAUDE.md — Standing Instructions for Claude Code

This file is automatically read by Claude Code at the start of every session. It is the project's standing operating agreement.

---

## Required Reading (every session, before writing any code)

1. **`README.md`** — product vision, build sequence, success criteria for each step.
2. **`docs/architecture.md`** — authoritative technical specification. When the README says *what*, this document says *how*.
3. **`docs/BUILD_STATUS.md`** — current project state: which steps are done, which is in progress, what's deferred, and the log of key decisions. This file is the single source of truth for "where are we?" when switching between Claude environments (Desktop, CLI, web, IDE).

**Do not write code before reading all three files in full.** If a task references "Step N" of the build sequence, locate that step in the README, then read the architecture section it cross-references before implementing. Cross-check `BUILD_STATUS.md` so you do not re-do already-completed work or skip a blocker.

---

## Working Agreement

### Execution model

- Work one build step at a time. Do not skip ahead.
- Before starting a step, restate the step's success criteria and the architecture sections it depends on.
- When a step is complete, verify the success criteria pass. Do not advance to the next step until they do.
- **Update `docs/BUILD_STATUS.md` at the end of every step** — flip the step's status to ✅, tick verified success criteria, update the "Current step" section at the top, add any new decisions to the log, and commit it with the step's code. This file keeps Claude Desktop, Claude CLI, and any other client in sync about project state.
- If you believe a step's spec is wrong or incomplete, stop and raise it — do not silently improvise.

### Authoritative sources (in order)

1. `docs/architecture.md` — for technical contracts (API surfaces, data layouts, ownership rules).
2. `README.md` — for scope, success criteria, and roadmap.
3. The non-negotiable rules in `docs/architecture.md` Section 15 — these override everything except explicit user instruction.

### Non-negotiable rules (from Section 15 of architecture.md)

1. The C++ shim stays under 2,000 lines across all `.cpp` and `.h` files in `shim/src/` and `shim/include/`.
2. No Rust-to-C++ exceptions. All errors translate at the ABI boundary into `BeaconStatus` codes.
3. No tensor copies on the MLX backend. Weights are mmap'd; KV cache allocated once at engine load.
4. No cross-backend tensors. One backend per engine instance, chosen at load time.
5. No blocking operations in the decode hot path. All I/O, allocation, and parsing happens at engine load.
6. Logit-level correctness tests pass before any optimization work.

If an approach would violate any of these, stop and flag it rather than proceeding.

### Code style

- Rust: enforce `rustfmt` defaults, `clippy::pedantic` with documented exceptions.
- C++ (shim only): C++20, clang-format, no exceptions crossing the ABI.
- All public APIs documented with doc comments. All non-trivial internal decisions documented with inline comments explaining *why*, not *what*.
- Tests first for correctness-critical code (format parsing, tokenizer, forward pass).

### Commits

This section applies to every Claude environment (Desktop, CLI, web, IDE). Follow it verbatim so history stays clean across handoffs.

- **One commit per logical change.** A logical change is one step's worth of work, or one infra/tooling change, or one doc change — never a mix.
- **Do not mix steps within a single commit.** If Step 2's work touched shared files (`.gitignore`, `.github/workflows/ci.yml`, `Cargo.toml`, etc.) that Step 1 also touched, stage only the Step-1 delta for Step 1's commit and the Step-2 delta for Step 2's commit. Use `git add -p` or targeted `git add <path>` plus on-the-fly edits if needed — do not shove the whole final file into the earlier commit.
- **Each commit must build standalone.** Anyone checking out any commit should get a tree where the step's success criteria pass. This keeps `git bisect` useful.
- **Commit messages:** imperative mood, reference the build step in the subject. Examples:
  - `Step 1: scaffold Cargo workspace with stub crates and CI`
  - `Step 2: add MLX shim C ABI and CMake build`
  - `docs: introduce BUILD_STATUS.md as cross-session state`
  - `ci: enable Metal Toolchain download on macos-14`
- **Update `docs/BUILD_STATUS.md` as part of the step's commit.** When a step completes, flip the status, tick verified criteria, update "Current step", and log new decisions — all in the same commit as the step's code. Exception: the one-time introduction of `BUILD_STATUS.md` itself is its own commit.
- **Never commit unless the user explicitly asks.** Do not commit at end-of-step by default; surface a summary and wait for "commit this" (or equivalent).
- **Do not skip hooks** (`--no-verify`, `--no-gpg-sign`) unless the user explicitly asks. If a hook fails, fix the underlying issue and make a new commit.
- **Do not amend published commits.** After a push, roll forward with a new commit rather than rewriting history.

### Tool and library choices

These are decided; do not propose alternatives without strong reason:

- **Rust ecosystem:** tokio (async), axum (HTTP), clap (CLI), serde (serialization), thiserror (errors), anyhow (app-level errors), memmap2 (mmap), bindgen (C FFI), cc/cmake crate (build integration).
- **Python bindings:** PyO3 + maturin.
- **Node bindings:** napi-rs.
- **C++ build:** CMake, MLX as git submodule at `shim/third_party/mlx`.
- **Testing:** built-in Rust test harness, criterion for benchmarks.

### Platform priority

1. macOS ARM64 (M-series) — primary, must always work.
2. Linux x86_64 — secondary, CPU backend must work.
3. Linux ARM64 — tertiary, CPU backend must work.
4. Windows x86_64 — build-only in v0.1, no optimization priority.
5. macOS x86_64 — build-only, deprecated priority.

### When to ask vs. when to decide

- **Decide autonomously:** implementation details within a spec'd contract, naming of internal types, test coverage strategy, minor refactors.
- **Ask the user:** changes to the public API, decisions that affect the v0.1 scope, trade-offs between spec'd requirements, anything that would touch the non-negotiable rules.

### Dependencies

- Prefer zero new runtime dependencies when a standard-library solution exists.
- When adding a crate, justify it in the PR description. Audit for maintenance status (commits in last 12 months, reasonable download count, compatible license).
- Never add a dependency with a non-permissive license (GPL, AGPL) without explicit user approval.

---

## Session start protocol

On every new session, your first response must:

1. Confirm you have read `README.md`, `docs/architecture.md`, and `docs/BUILD_STATUS.md`.
2. State the current build step (per `docs/BUILD_STATUS.md` → "Current step").
3. State the specific task the user is asking you to perform.
4. Then proceed with the task.

If any of (1)-(3) is unclear, ask before writing code. If `docs/BUILD_STATUS.md` is missing or stale (e.g., its "Current step" does not match what the working tree and `git log` suggest), flag the discrepancy before proceeding — treat the codebase and git history as ground truth and ask the user how to reconcile.

---

## Escalation

If you encounter any of the following, stop and escalate to the user:

- A spec that contradicts itself across the two documents.
- A requirement that cannot be implemented without violating a non-negotiable rule.
- An external dependency (model weights, API, library) that is unavailable or has changed significantly since the spec was written.
- A step's success criteria that you cannot verify in the current environment (e.g., needs hardware you don't have access to).

Do not silently work around these.
