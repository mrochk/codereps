# codereps

This repository contains resources and code used during my 2024 summer internship at the Bordeaux Computer Science Laboratory (LaBRI).

My work aimed at finding ways to improve LLMs performances and capacities on software engineering tasks by exploring the use of **efficiently tokenizable intermediate code representations**.

### *Phase 1*

The first step was to explore different code representations, that phase gave birth to: 
- A Python package for Control Flow Graphs serialization: [funcskeleton](https://github.com/mrochk/funcskeleton) 
- A dataset created from [*CodeSearchNet*](https://huggingface.co/datasets/code-search-net/code_search_net) available on [my HuggingFace page](https://huggingface.co/mrochk) containing source code with its corresponging AST & CFG serializations. 

## *Phase 2*

Then, I started training small transformer models from scratch on a dummy binary classification task using my previously created dataset: exchanging randomly code representations in the dataset, and training the model to find the correct relation between given pairs of (SRC, AST), (SRC, CFG) or (AST, CFG) that is to output 0 if they are relayed, 1 otherwise. These experiments can be found in the folder [from_scratch](./from_scratch).

## *Phase 3*

I fine-tuned and extended the pre-trained transformer encoder *DistilBERT* on the same task and dataset, one version of the model was simply fine tuned by adding a classification head on top of it, and another one was extended by adding to the model a new custom embedding layer using our own tokenizer, both models demonstrated similar performance. The code for this can be found on Kaggle.