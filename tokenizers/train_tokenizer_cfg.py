from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

dataset = load_dataset('mrochk/src_cfg_ast')

train = dataset['train']

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

from tokenizers.pre_tokenizers import Split, Sequence

behavior = 'isolated'

pretokenizer = Sequence([
    Split('[SEP1]', behavior=behavior),
    Split('[SEP2]', behavior=behavior),
    Split('[SEP3]', behavior=behavior),
])

tokenizer.pre_tokenizer = pretokenizer

cfg = "[5][SEP1][If][If][Return][None][For][If][If][None][Return][SEP2][1,2][3,2][][4][5,6][7,4][8,2][4][][SEP3][0][2][0][1][0][1][1][0][0]"
print(pretokenizer.pre_tokenize_str(cfg))

for vocab_size in [500, 1000, 2000, 5000]:

    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    tokenizer.train_from_iterator(train['cfg'], trainer, len(train['cfg']))

    tokenizer.save(f'tokenizer_cfg[{vocab_size}].json')

print(tokenizer.encode(cfg).tokens)