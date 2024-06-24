# Tokenizers
This folder contains the [code](./train_tokenizers.py) used to train tokenizers for encoding both ASTs and CFGs.

**To use the tokenizers without having to re-train them, import them from the [tokenizers](./tokenizers/) directory.**

We use HuggingFace's [`tokenizers`](https://github.com/huggingface/tokenizers) library for training.

Both trained tokenizers are BPEs ([*Byte pair encoding*](https://en.wikipedia.org/wiki/Byte_pair_encoding)), with variations for different vocabsize $\in \{500, 1000, 2000, 5000\}$.

*For the source code we use a pre-trained tokenizer.*