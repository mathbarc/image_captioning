import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.ops import Conv2dNormActivation


def create_encoder(embed_size, dropout = 0.2, pretrained=True):
    backbone = models.mobilenet_v3_large(models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    for param in backbone.parameters():
        param.requires_grad_(not pretrained)
    
    modules = list(backbone.children())
    cnn = modules[0]
    

    cnn.add_module("conv_output", Conv2dNormActivation(cnn[-1].out_channels, embed_size,activation_layer=None))
    cnn.add_module("activation", nn.Tanhshrink())
    cnn.add_module("pool", nn.AdaptiveAvgPool2d(1))
    cnn.add_module("flatten",nn.Flatten())
    cnn.add_module("dropout",nn.Dropout(dropout))
    
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

        # self.output_activation = nn.LogSoftmax(-1)
    
    def forward(self, features, captions, hidden):
        captions = captions[:,:-1]

        embeds = self.embed(captions)
        inputs = torch.cat((features,embeds), dim=1)

        return self._forward(inputs, hidden)
    
    def _forward(self, inputs, hidden):

        x, hidden = self.rnn(inputs, hidden)

        x = self.dropout(x)

        x = self.linear(x)

        # x = self.output_activation(x)

        return x, hidden

    def repackage_hidden(self, h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

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
        self.encoder = create_encoder(embed_size, dropout, pretreined)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, dropout)
    
    def forward(self, images, captions, hidden):
        features = self.encoder(images).unsqueeze(dim=1)
        return self.decoder(features, captions, hidden)
    
    def sample(self, image, max_len=20):
        features = self.encoder(image).unsqueeze(dim=1)
        captions = self.decoder.sample(features, max_len)
        return captions



def get_transform():
    return transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Resize(480,antialias=True),
        transforms.RandomHorizontalFlip(), 
        #transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])

def get_inference_transform():
    return transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Resize(480,antialias=True),
        #transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])

if __name__=="__main__":
    cnn = ImageCaptioner(1024, 1024, 4376, 3)

    print(cnn)

    
