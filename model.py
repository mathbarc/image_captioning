from os import stat
import torch
import torch.nn as nn
import torchvision.models as models
import numpy


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size:int, hidden_size:int, vocab_size:int, num_layers:int=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, features, captions):
        if(features.size()[1]==self.embed_size):
            x = features
        else:
            x = self.embed(features)
        cap = self.embed(captions)
        x, captions = self.rnn(x, cap)
        x = nn.functional.softmax(self.linear(x))
        return x, captions

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of token ids of length max_len) "
        if states is None:
            states = self.init_hidden(1)
        features = inputs
        tokens = []
        for i in range(max_len):
            outVector, states = self.forward(features, states)
            tokens.append(numpy.argmax(outVector))

        return tokens


    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return h, c
