# Model weights

Place each local model in its own directory here (any Hugging Face snapshot layout works):

```
models/
  DeepSeek-R1-Distill-Qwen-14B/   https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
  DeepSeek-R1-Distill-Qwen-32B/   https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
  QwQ-32B-Preview/                https://huggingface.co/Qwen/QwQ-32B-Preview
  Qwen3-32B/                      https://huggingface.co/Qwen/Qwen3-32B
```

For example:

```
pip install -U "huggingface_hub[cli]"
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local-dir models/DeepSeek-R1-Distill-Qwen-32B
```

The paper ran the 14B model in bf16 and the 32B models in 4-bit NF4 (`--load-4bit`; needs
`bitsandbytes`). Weights are not part of this repository (`models/*` is git-ignored except this file).
