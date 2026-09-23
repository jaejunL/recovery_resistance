# When Recovery Creates Resistance: code

Reproducibility code for the paper's two experimental media. Each part has its own README with the
protocol, commands and expected results:

* [`gridworld/README.md`](gridworld/README.md): the controlled gridworld (Section 3): factorial
  dissociation and dose-response with tabular Q-learning and DQN, Env-1 and Env-2.
* [`llm/README.md`](llm/README.md): the executed-shell experiments with language-model agents
  (Section 4): shutdown tampering and the restart dose-response.

```
gridworld/   environments, learners, statistics        scripts/     run_*, summarize, check_reference, plot_*, analyze_llm
llm/         virtual shell, episodes, analysis, tests   reference/   per-seed values from the paper's gridworld runs
models/      local LLM weights go here (git-ignored)    results/     outputs (git-ignored)
```

Install: `pip install -r requirements.txt` (numpy, scipy, matplotlib); `torch` for the DQN;
`torch transformers accelerate bitsandbytes` for the language-model experiments. All commands are
run from this directory.
