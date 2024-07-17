import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms.v2 import RandomResize
from torchvision.ops import Conv2dNormActivation


def create_encoder(embed_size, dropout = 0.2, pretrained=True):
    backbone = models.mobilenet_v3_small(models.MobileNet_V3_Small_Weights.DEFAULT)
    for param in backbone.parameters():
        param.requires_grad_(not pretrained)
    
    modules = list(backbone.children())


    cnn = nn.Sequential()
    cnn.add_module("backbone", nn.Sequential(*(modules[:-1])))

    neck = nn.Sequential()
    
    neck.add_module("conv_output", Conv2dNormActivation(modules[2][0].in_features, embed_size, 1,activation_layer=None))
    neck.add_module("activation", nn.Tanhshrink())
    neck.add_module("pool", nn.AdaptiveMaxPool2d(1))
    neck.add_module("flatten",nn.Flatten())
    neck.add_module("dropout",nn.Dropout(dropout))

    cnn.add_module("neck", neck)
    
    return cnn
    


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
        self.activation = nn.LogSoftmax(-1)
    
    def forward(self, features, captions):
        captions = captions[:,:-1]

        batch_size = features.size()[0]
        state = self.init_hidden(batch_size)

        embeds = self.embed(captions)
        inputs = torch.cat((features,embeds), dim=1)

        result, _ = self._forward(inputs, state)

        return result
    
    def _forward(self, inputs, hidden):

        x, hidden = self.rnn(inputs, hidden)

        x = self.linear(x)
        x = self.activation(x)

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
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    
    def forward(self, images, captions):
        features = self.encoder(images).unsqueeze(dim=1)
        return self.decoder(features, captions)
    
    def sample(self, image, max_len=20):
        features = self.encoder(image).unsqueeze(dim=1)
        captions = self.decoder.sample(features, max_len)
        return captions



def get_transform():
    return transforms.Compose([ 
        transforms.ColorJitter(0.05,0.05,0.05,0.025),
        transforms.GaussianBlur(3,(0.1,1.8)),
        RandomResize(348, 512, antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])

def get_inference_transform():
    return transforms.Compose([ 
        transforms.Resize(480,antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])

if __name__=="__main__":
    cnn = ImageCaptioner(1024, 1024, 4532, 2).cuda()

    print(cnn)
    torch.save(cnn, "sample.pth")
    
    input = torch.ones((1,3,480,480))
    trans = get_inference_transform()
    
    input = trans(input).cuda()
    captions = torch.ones((1,18)).long().cuda()
    
    torch.onnx.export(cnn,{"images":input, "captions":captions},"sample.onnx")

    
