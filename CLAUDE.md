# Rules for Claude / agents in this repo

## Git
- Never `git commit`, `git push`, or create branches/PRs. Leave changes in the working tree; the owner commits.
- No `Co-Authored-By` or any Claude attribution anywhere (files, messages, notebooks).

## Code
- Shared tools live in the package. Each tool is small, has typed inputs and outputs written in its docstring, and does one thing.
- Anything that takes more than a minute goes through `cache` (keyed on its parameters and input file hashes). A rerun with the same parameters must not recompute.
- Heavy work lives in a function (and a `scripts/` entry point for batch runs). Notebooks call it; they do not contain it.
- Do not change physics while moving code. Moving code needs a regression check against the old function's output.

## Notebooks
- Run top to bottom, in their numbered order. One task per notebook.
- One purpose per cell. A short markdown cell before each code cell says what it does and why, in plain sentences.
- Tone: written by a physicist for colleagues. No emoji, no "Let's", no bold on every other word, no summary table after every step, no "Key takeaways", no long dash chains.
- Show plots in full. Collapse output only when it is extremely long.
- Save every figure with `uuls.figures.FigureSaver(study, notebook).save(fig, name)`, which writes `outputs/figures/<study>/<notebook>/<name>.{png,pdf}`.
- Every notebook states its inputs (config, cache keys) at the top and its outputs at the bottom.
- Commit notebooks with outputs (the owner wants to read results on GitHub). Check the JSON is valid (no merge markers) before finishing.

## Environments
- Main: `env/uncertain-uls.yml` (gammapy 2.1, ligo.skymap, tilepy, emcee). Python at `$UULS_PYTHON` or the `uncertain-uls` conda env.
- simulst runs in its own env (gammapy 1.3) and is only ever called as a subprocess through `uuls.simdata`. Never import simulst from the main env.
- Machine has 2 CPUs. Keep test and notebook defaults small (seconds to minutes); large runs go through `scripts/` for the cluster.

## Layout
- `uuls/`: shared tools. `studies/<n>_<name>/`: numbered notebooks + `configs/`. `configs/cases/`: the shared mock cases.
- `data/` (inputs), `cache/` (keyed results), `outputs/` (figures, results): all ignored by git except small tracked inputs listed in `.gitignore`.
- Old reference implementation, read-only: `../gw-gamma-global-uls/gwuls/`. Port physics from it; do not import it.
