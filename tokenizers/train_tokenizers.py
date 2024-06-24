from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Split, Sequence

PATH_TO_TOKENIZERS = './tokenizers/'

# import dataset
dataset = load_dataset('mrochk/src_ast_cfg')

# we train on "train" split
train = dataset['train']

# defining our BPE tokenizers
tokenizer_ast = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer_cfg = Tokenizer(BPE(unk_token="[UNK]"))

# for the ast pretokenizer we simply split line by line
# (AST = "MODULE_FUNCTIONDEF_..._")
pretokenizer_ast = Whitespace()

# we keep the separator tokens, we simply isolate them
behavior = 'isolated'

# for the cfg pretokenizer we split each sentence 
# using our separators
pretokenizer_cfg = Sequence([
    Split('[SEP1]', behavior=behavior),
    Split('[SEP2]', behavior=behavior),
    Split('[SEP3]', behavior=behavior),
])

tokenizer_ast.pre_tokenizer = pretokenizer_ast
tokenizer_cfg.pre_tokenizer = pretokenizer_cfg

length = len(train['ast'])

# we produce 4 variations for both with different vocabulary size
for vocab_size in [500, 1000, 2000, 5000]:

    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    tokenizer_ast.train_from_iterator(train['ast'], trainer, length)
    tokenizer_cfg.train_from_iterator(train['cfg'], trainer, length)

    tokenizer_ast.save(f'{PATH_TO_TOKENIZERS}tokenizer_ast_{vocab_size}.json')
    tokenizer_ast.save(f'{PATH_TO_TOKENIZERS}tokenizer_cfg_{vocab_size}.json')