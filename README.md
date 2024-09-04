# codereps

This repository contains resources and code used during my 2024 summer internship at the Bordeaux Computer Science Laboratory (LaBRI).

My work aimed at finding ways to improve LLMs performances and capacities on software engineering tasks by exploring the use of **efficiently tokenizable intermediate code representations**.

## *Phase 1*

The first step was to explore different code representations, that phase gave birth to: 
- A Python package for Control Flow Graphs serialization: [funcskeleton](https://github.com/mrochk/funcskeleton) 
- A dataset created from [*CodeSearchNet*](https://huggingface.co/datasets/code-search-net/code_search_net) available on [my HuggingFace page](https://huggingface.co/mrochk) containing source code with its corresponging AST & CFG serializations. 

## *Phase 2*

Then, I started training small transformer models from scratch on a dummy binary classification task using my previously created dataset: exchanging randomly code representations in the dataset, and training the model to find the correct relation between given pairs of (SRC, AST), (SRC, CFG) or (AST, CFG) that is to output 0 if they are relayed, 1 otherwise. These experiments can be found in the folder [from_scratch](./from_scratch).

## *Phase 3*

I fine-tuned and extended the pre-trained transformer encoder *DistilBERT* on the same task and dataset, one version of the model was simply fine tuned by adding a classification head on top of it, and another one was extended by adding to the model a new custom embedding layer using our own tokenizer, both models demonstrated similar performance. The code for this can be found on Kaggle.

***
*A talk that I gave **in french** after completing the first 3 phases of this project was recorded and is available [here](https://drive.google.com/file/d/1P4517oADcLtzRxU3f12o0lRi4WYL9_49/view?usp=sharing).* 
***

## *Phase 4*

This is where I am now, and this phase consist of the following experiment: *Take a text/code-generation model and a dataset of Python problems, with solutions such that we can use a metric to evaluate the model's performance, and train & eval 3 versions of it:*

- One version that we simply fine-tune on the original dataset of $(\text{prompt}, \text{solution})$. ($A$)
- Another that we fine-tune on $((\text{prompt}, \text{solution IR*}), \text{solution})$. ($B$) *(IR = Intermediate Representation)
- Finally, we extend the model to add new vocabulary and embedddings of the IR tokenizer and fine-tune the same way as $B$. ($C$)

*It is possible that for the last version, we may need to first train him on a larger dataset so that the model can learn the meaning of the new added embeddings. We could use my dataset available on HuggingFace for that.*

Let $\text{perf}(M)$ denote the result of some version of the model evaluated using some chosen metric, on some unseen part of the fine-tuning dataset, an interesting / ideal result for us would be to see 
$$\text{perf}(A) < \text{perf}(B)\;\text{and}\;\text{perf}(B) \approx \text{perf}(C)$$

Thus showing that $B$ learned the IR meaning or at least some part of it, and that $C$ did the same, but with a much more efficient tokenization scheme and new embeddings learned from scratch. 

### *Resources*

**Models**

- codegen https://huggingface.co/Salesforce/codegen-350M-mono  
- santacoder https://huggingface.co/bigcode/santacoder 
- codebert https://arxiv.org/abs/2002.08155
- codeparrot https://huggingface.co/codeparrot/codeparrot
- incoder https://huggingface.co/facebook/incoder-1B
- polycoder https://huggingface.co/NinedayWang/PolyCoder-0.4B
- phi-1 https://huggingface.co/microsoft/phi-1
- refact https://huggingface.co/smallcloudai/Refact-1_6B-fim

*A priori* we are going to use a model in the family of Phi models.

**Datasets**

- APPS https://github.com/hendrycks/apps
- MBPP https://github.com/google-research/google-research/tree/master/mbpp

**Metrics**

- pass@k https://arxiv.org/pdf/2107.03374
- codescore https://arxiv.org/abs/2301.09043
