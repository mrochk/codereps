# Data Preparation

[`create_dataset.py`](./create_dataset.py) contains the code I used to prepare preprocess the data.

***If you do not want to run this script yourself (it can take a lot of time) I uploaded the dataset at https://huggingface.co/datasets/mrochk/src_ast_cfg.***

*You can load it simply using:*
``` python
from datasets import load_dataset
dataset = load_dataset('mrochk/src_cfg_ast')
```

Creating the dataset for this project can be described with the following steps:
1. Load the [CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net) *Python* subset dataset.
2. Keep only "*single simple*" functions (remove samples containing nested functions or classes, as well as functions taking special arguments such as `**kwargs` for example).
3. For each sample, build its corresponding *AST* (using `get_ast_nodes_dfs` function) and *CFG* (using [`funcskeleton`](https://github.com/mrochk/funcskeleton)).
4. Save the 3 splits.

At the we are left with:
- `train.json`      (278620 samples)
- `test.json`       (15169 samples)
- `validation.json` (15566 samples)
