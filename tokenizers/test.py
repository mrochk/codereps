from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file('tokenizers/tokenizer_cfg_500.json')
print(tokenizer.get_vocab_size())
print(tokenizer.encode('[0][SEP1][If]').tokens)
