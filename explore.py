import regex as re
from bpe import Tokenizer, GPT2_SPLIT

text = open("novel.txt", encoding="utf-8").read()
tok = Tokenizer()
tok.train(text, 1000)

# 1. Which learned tokens contain any digit?
print([tok.vocab[i] for i in range(256, 1000) if any(48 <= b <= 57 for b in tok.vocab[i])])

# 2. What chunks does the regex hand to BPE?
print(re.findall(GPT2_SPLIT, "12345 + 678"))