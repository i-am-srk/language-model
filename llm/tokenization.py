import re
from importlib.metadata import version
import tiktoken
import torch

with open(file='the-verdict.txt', mode='r', encoding='utf-8') as f:
    raw_text = f.read()

# all tokens
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# unique tokens
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# token IDs
vocab = {token: integer for integer, token in enumerate(all_tokens)}

# simple tokenizer
class SimpleTokenizer:
    def __init__(self, vocabulary):
        """
        param
        :vocabulary - type: dict[str, int]
        """
        self.str_to_int = vocabulary
        self.int_to_str = {i: s for s, i in vocabulary.items()}

    def encode(self, text):
        """
        param
        :text - type: str
        
        return
        : list[int]
        """
        preprocessed = re.split(r'([,.;:?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        tokens = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        token_ids = [self.str_to_int[token] for token in tokens]
        return token_ids

    def decode(self, token_ids):
        """
        param
        :token_ids - type: list[int]
        
        return
        : str
        """
        text = " ".join([self.int_to_str[ids] for ids in token_ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))
# print(f"text: {text}")

# tokenizer = SimpleTokenizer(vocab)
# print(f"encoded text: {tokenizer.encode(text)}")

# print(f"decoded text: {tokenizer.decode(tokenizer.encode(text))}")

# BPE
# print(f"toktoken version: {version('tiktoken')}")
tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)
# print(f"{len(enc_text)}")

# pytorch
# print(f"{version('torch')}")
# print(f"pytorch acceleration availability: {torch.backends.mps.is_available()}")

tensor0d = torch.tensor(1)
print(tensor0d)
tensor1d = torch.tensor([1, 2, 3])
print(tensor1d)
tensor2d = torch.tensor([[1, 2, 3],[4, 5, 6]])
print(tensor2d)
tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(tensor3d)