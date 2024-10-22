import re

with open(file='the-verdict.txt', mode='r', encoding='utf-8') as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

class SimpleTokenizer:
    def __init__(self, vocabulary):
        """
        :param
        :type vocabulary: dict{str : int}
        """
        self.str_to_int = vocabulary
        self.int_to_str = {i: s for s, i in vocabulary.items()}

    def encode(self, text):
        """
        :param
        :type text: str
        :return: list[int]
        """
        preprocessed = re.split(r'([,.;:?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        tokens = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        token_ids = [self.str_to_int[token] for token in tokens]
        return token_ids

    def decode(self, token_ids):
        """
        :param
        :type token_ids: list[int]
        :return: str
        """
        text = " ".join([self.int_to_str[ids] for ids in token_ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizer(vocab)
print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))
