import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        # word2idx["home"] = 9
        self.idx2word = []
        # idx2word[9] = "home"

    def add_word(self, word):
    # update both word2idx and idx2word
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        # equal to len(self.word2idx)
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        # "Token" means numerical representitive of word.
        # "Tokenize" a sentense means to map words in sentense to word id
        # which is unique to the word.
        # ex. "a cat is a pet".tokenize() -> "1 2 3 1 4"
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                # "a cat is a pet<eos>"
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        # word2idx = {"A": 1, "cat": 2, "is": 3, "pet": 3, "<eos>": 4}
        # idx2word = ["A",    "cat",    "is",    "pet"   , "<eos>"   ]
        # tokens = 6
        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            # ids = [0, 0, 0, 0, 0, 0]
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1 
        # ids = [1, 2, 3, 1, 4, 5]
        return ids
