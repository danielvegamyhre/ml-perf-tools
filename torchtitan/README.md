# torchtitan memory analysis

This script can be used to analyze memory usage of different layers in a
torchtitan llama3 model with different combinations of configurations
(i.e., torch.compile, FSDP2, activation checkpointing settings, etc.)

### usage

1. [Install](https://github.com/pytorch/torchtitan?tab=readme-ov-file#installation) torchtitan.
2. Copy this script into the torchtitan root directory.
3. Run `python memory_analysis.py -h` to see the full list of parameters.
