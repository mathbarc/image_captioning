import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms.v2 import RandomResize
from torchvision.ops import Conv2dNormActivation


def create_encoder(embed_size, dropout = 0.2, pretrained=True):
    if pretrained:
        backbone = models.mobilenet_v3_small(models.MobileNet_V3_Small_Weights.DEFAULT)
        for param in backbone.parameters():
            param.requires_grad_(not pretrained)
    else:
        backbone = models.mobilenet_v3_small()
        
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
    
    
class MobileNetV3Backbone(nn.Module):
    def __init__(self, embed_size, pretrained=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        if pretrained:
            backbone = models.mobilenet_v3_small(models.MobileNet_V3_Small_Weights.DEFAULT)
            for param in backbone.parameters():
                param.requires_grad_(not pretrained)
        else:
            backbone = models.mobilenet_v3_small()
        
        modules = list(backbone.children())

        self.cnn = nn.Sequential()
        self.cnn.add_module("backbone", nn.Sequential(*(modules[:-1])))

        self.neck = ImageCaptionerNeck(modules[2][0].in_features, embed_size)
    
    def forward(self, image):
        x = self.cnn(image)
        x = self.neck(x)
        return x
    
class EfficientNetV2Backbone(nn.Module):
    def __init__(self, embed_size, pretrained=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        backbone = models.efficientnet_v2_s(models.EfficientNet_V2_S_Weights.DEFAULT)
        for param in backbone.parameters():
            param.requires_grad_(not pretrained)
        
        modules = list(backbone.children())

        self.cnn = nn.Sequential()
        self.cnn.add_module("backbone", nn.Sequential(*(modules[:-1])))

        self.neck = ImageCaptionerNeck(modules[2][-1].in_features, embed_size)
    
    def forward(self, image):
        x = self.cnn(image)
        x = self.neck(x)
        return x
        
    
    
class ImageCaptionerBackbone(nn.Module):
    def __init__(self, embed_size:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.conv1 = Conv2dNormActivation(3, 16, (3, 3), (1, 1))
        self.conv2 = Conv2dNormActivation(16, 32, (3, 3), (1, 1))
        self.conv3 = Conv2dNormActivation(32, 64, (3, 3), (1, 1))
        self.conv4 = Conv2dNormActivation(64, 128, (3, 3), (1, 1))
        self.conv5 = Conv2dNormActivation(128, 256, (3, 3), (1, 1))
        self.conv6 = Conv2dNormActivation(256, 512, (3, 3), (1, 1))
        self.conv7 = Conv2dNormActivation(512, 1024, (3, 3), (1, 1))
        
        self.pool1 = torch.nn.MaxPool2d(2, stride=2)
        self.pool2 = torch.nn.MaxPool2d(2, stride=1)
        
        self.neck = ImageCaptionerNeck(1024,embed_size)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.pool1(x)

        x = self.conv5(x)
        x = self.pool1(x)

        x = self.conv6(x)
        x = self.pool2(x)
        
        x = self.conv7(x)
        x = self.neck(x)
        
        return x
    
        

class ImageCaptionerNeck(nn.Module):
    def __init__(self, input_features:int, embed_size:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.conv_output = Conv2dNormActivation(input_features, embed_size, 1,activation_layer=None)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
    
    def forward(self, input):
        x = self.conv_output(input)
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.unsqueeze(x,1)
        
        return x


class ImageCaptionerHead(nn.Module):
    def __init__(self, embed_size:int, hidden_size:int, vocab_size:int, num_layers:int=1, dropout:float=0.5):
        super(ImageCaptionerHead, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first=True, dropout=dropout)

        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.activation = nn.LogSoftmax(-1)
    
    
    
    def forward(self, inputs, hidden):

        x, hidden = self.lstm(inputs, hidden)

        x = self.linear(x)
        x = self.activation(x)

        return x, hidden

    def repackage_hidden(self, h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

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
    def __init__(self, backbone_type:str, embed_size:int, hidden_size:int, vocab_size:int, num_layers:int=1, dropout=0.2, max_len=20) -> None:
        super().__init__()
        
        if backbone_type == "mobilenet":
            self.encoder = MobileNetV3Backbone(embed_size)
        elif backbone_type == "efficientnet_v2":
            self.encoder = EfficientNetV2Backbone(embed_size)
        else:
            self.encoder = ImageCaptionerBackbone(embed_size)
        self.decoder = ImageCaptionerHead(embed_size, hidden_size, vocab_size, num_layers, dropout)
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.max_len = max_len
    
    def forward(self, images, check_finished=True):
        features = self.encoder(images)
        
        tokens = None
        batch_size = images.size()[0]
        states = self.decoder.init_hidden(batch_size)
        
        prediction, states = self.decoder(features, states)
        _, max_idx = torch.max(prediction, dim=2)
        token = max_idx
        tokens = token
        
        count = 0
        
        while count < self.max_len and ((max_idx != 1).all() or not check_finished):
            prediction = self.embed(max_idx)
            prediction, states = self.decoder(prediction, states)
            
            _, max_idx = torch.max(prediction, dim=2)
            token = max_idx
            tokens = torch.cat([tokens, token],dim=-1)
            count +=1
            
        return tokens
    
    def compute_gradients(self, images, captions):
        features = self.encoder(images)
        captions = captions[:,:-1]

        batch_size = features.size()[0]
        state = self.decoder.init_hidden(batch_size)

        embeds = self.embed(captions)
        inputs = torch.cat((features,embeds), dim=1)

        result, _ = self.decoder.forward(inputs, state)

        return result
    
    def sample(self, image, max_len=20):
        features = self.encoder(image).unsqueeze(dim=1)
        captions = self.decoder.sample(features, max_len)
        return captions

    def save(self, name, save_subcomponents=False):
        
        
        input = torch.ones((1,3,480,480))
        trans = get_inference_transform()
        
        input = trans(input).cuda()
        
        if not save_subcomponents:
        
            torch.onnx.export(
                self,
                (input,False),
                f"{name}.onnx",
                input_names=["images"],
                output_names=["output"],
                opset_version=11,
                dynamic_axes={"images":{0:"batch_size"}, 
                            "output":{0:"batch_size"}}
            )
        
        else:
            torch.onnx.export(
                self.encoder,
                input,
                f"{name}_encoder.onnx",
                input_names=["images"],
                output_names=["output"],
                opset_version=11,
                dynamic_axes={"images":{0:"batch_size"}, 
                            "output":{0:"batch_size"}}
            )
            
            word = torch.ones((1,1)).long().cuda()
            
            torch.onnx.export(
                self.embed,
                word,
                f"{name}_embed.onnx",
                input_names=["word"],
                output_names=["embed"],
                opset_version=11,
                dynamic_axes={"word":{0:"batch_size"}, 
                            "embed":{0:"batch_size"}}
            )
            
            embed = torch.ones((1,1,self.embed_size)).cuda()
            hidden = self.decoder.init_hidden(1)
            
            torch.onnx.export(
                self.decoder,
                (embed, hidden),
                f"{name}_decoder.onnx",
                input_names=["embed", "state_input_h", "state_input_c"],
                output_names=["output","state_output_h", "state_output_c"],
                opset_version=11,
                dynamic_axes={"embed":{0:"batch_size"}, 
                              "state_input_h":{1:"batch_size"}, 
                              "state_input_c":{1:"batch_size"}, 
                              "state_output_h":{1:"batch_size"},
                              "state_output_c":{1:"batch_size"},
                              "output":{0:"batch_size"}}
            )
        

def get_transform():
    return transforms.Compose([ 
        # transforms.ColorJitter(0.05,0.05,0.05,0.025),
        # transforms.GaussianBlur(3,(0.1,1.8)),
        RandomResize(348, 512, antialias=True),
        # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])

def get_inference_transform():
    return transforms.Compose([ 
        transforms.Resize(480,antialias=True),
        # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])


def jaccard_index():
    pass

if __name__=="__main__":
    cnn = ImageCaptioner(256, 128, 4532, 1).cuda()

    print(cnn)
    cnn.save("test")

    
