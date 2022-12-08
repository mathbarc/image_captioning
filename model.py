from os import stat
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


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
        self.rnn = nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        # self.out_activation = nn.Softmax(self.vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:,:-1]
        batch_size = features.size()[0]
        state = self.init_hidden(batch_size)
        embeds = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(dim=1),embeds), dim=1)
        
        x, state = self.rnn(inputs, state)
        x = self.linear(x)
        # x = self.out_activation(x)
        
        return x

    def sample(self, inputs, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of token ids of length max_len) "
        
        tokens = []
        batch_size = inputs.size()[0]
        states = self.init_hidden(batch_size)
        
        
        for i in range(max_len):
            outVector, states = self.rnn(inputs, states)
            outVector = self.linear(outVector)
            # outVector = self.out_activation(outVector)
            _, max_idx = torch.max(outVector, dim=2)
            token = max_idx.cpu().numpy()[0].item()
            tokens.append(token)
            if max_idx == 1:
                break

            inputs = self.embed(max_idx)
            # inputs = inputs.unsqueeze(1)
            

        return tokens


    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        if torch.cuda.is_available():
            h = h.to("cuda")
            c = c.to("cuda")
        
        return h,c

def get_transform():
    return transforms.Compose([ 
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                            (0.229, 0.224, 0.225))])