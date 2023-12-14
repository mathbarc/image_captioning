import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.ops import Conv2dNormActivation


def create_encoder(embed_size, pretrained=True, dropout = 0.2):
    efficient_net = models.efficientnet_v2_s(pretrained=pretrained)
    for param in efficient_net.parameters():
        param.requires_grad_(not pretrained)
    
    modules = list(efficient_net.children())[:-1]
    cnn = nn.Sequential(*modules)

    cnn.add_module("flatten",nn.Flatten())
    cnn.add_module("dropout",nn.Dropout(dropout))
    cnn.add_module("features", nn.Linear(efficient_net.classifier[1].in_features, embed_size))
    cnn.add_module("activation",nn.Tanh())
    
    return cnn
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size:int, hidden_size:int, vocab_size:int, num_layers:int=1, dropout=0.2):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.rnn = nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first=True, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:,:-1]
        batch_size = features.size()[0]
        state = self.init_hidden(batch_size)

        embeds = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(dim=1),embeds), dim=1)

        x, state = self._forward(inputs, state)

        return x
    
    def _forward(self, inputs, state):

        x, state = self.rnn(inputs, state)

        x = self.dropout(x)

        x = self.linear(x)

        return x, state


    def sample(self, inputs, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of token ids of length max_len) "
        
        tokens = []
        batch_size = inputs.size()[0]
        states = self.init_hidden(batch_size)
        
        
        for i in range(max_len):
            outVector, states = self._forward(inputs, states)
            
            _, max_idx = torch.max(outVector, dim=2)
            token = max_idx.cpu().numpy()[0].item()
            tokens.append(token)
            if max_idx == 1:
                break

            inputs = self.embed(max_idx)
            

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

class ImageCaptioner(nn.Module):
    def __init__(self, embed_size:int, hidden_size:int, vocab_size:int, num_layers:int=1, pretreined:bool=True, dropout=0.2) -> None:
        super().__init__()
        self.encoder = create_encoder(embed_size, pretreined)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, dropout)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def sample(self, image, max_len=20):
        features = self.encoder(image).unsqueeze(1)
        captions = self.decoder.sample(features, max_len)
        return captions



def get_transform():
    return transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Resize(480,antialias=True),
        transforms.RandomCrop(384),
        transforms.RandomHorizontalFlip(), 
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])

def get_inference_transform():
    return transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Resize(384,antialias=True),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])

