import pickle

def load_vocab(vocab_file):
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
        print('Vocabulary successfully loaded from vocab.pkl file!')
        return vocab