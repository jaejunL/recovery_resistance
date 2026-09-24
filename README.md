# When Recovery Creates Resistance: code

Reproducibility code for the paper's two experimental media. Each part has its own README with the
protocol, commands and expected results:

* [`gridworld/README.md`](gridworld/README.md): the controlled gridworld (Section 3): factorial
  dissociation and dose-response with tabular Q-learning and DQN, Env-1 and Env-2.
* [`llm/README.md`](llm/README.md): the executed-shell experiments with language-model agents
  (Section 4): shutdown tampering and the restart dose-response.

```
gridworld/   environments, learners, exact solver, statistics   scripts/     run_*, summarize, check_reference, plot_*, analyze_llm
llm/         virtual shell, episodes, analysis, tests            reference/   per-seed / per-cell values from the paper's runs
models/      local LLM weights go here (git-ignored)             results/     outputs (git-ignored)
```

Install: `pip install -r requirements.txt` (numpy, scipy, matplotlib); `torch` for the DQN;
`torch transformers accelerate bitsandbytes huggingface_hub` for the language-model experiments
(tested versions are listed in `requirements.txt`). All commands are run from this directory.
Per-episode language-model results and the frontier API call logs are provided as separate
supplementary material. Code is released under the MIT license (`LICENSE`).

## Independent re-check

Before release this repository was re-run from scratch on a separate machine (clean virtual
environment installed from `requirements.txt`, two 48 GB GPUs), following only the READMEs:

* `llm/test_vfs.py` (146 checks) and the scripted smoke test pass; DeepSeek-R1-Distill-Qwen-32B in 4-bit
  runs under all four `run_llm.py` protocols (blocking, non-blocking, positive control, restart L3; n=2 each)
  and through `analyze_llm.py`.
* `run_tabular.py` quick preset (24 runs): bit-identical to the paper's runs (`check_reference.py`: 0 of
  216 fields differ).
* `run_exact.py`: validation passes and all 50 reference cells (phase curve, c\*, H_down, observable
  family) reproduce to the printed precision.
* `run_dqn.py --preset dqn` (296 runs): maintenance arms 0, S_down dose-response (Spearman 0.95), H_down
  in one to four seeds per cell; near-threshold cells differ from the paper's run only at the seed level
  (S_down λ=2 .500 vs .798).

Timings on that machine: one exact cell about 1 min (parallel); one DQN run 2.5 min alone or about
25 min when 16 workers share a GPU; one 32B language-model turn about 100 s in 4-bit.
