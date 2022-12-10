import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms

from data_loader import get_loader
from model import EncoderCNN, DecoderRNN, get_transform
import mlflow


## TODO #1: Select appropriate values for the Python variables below.
batch_size = 64          # batch size
vocab_threshold = 5        # minimum word count threshold
vocab_from_file = True    # if True, load existing vocab file
embed_size = 800           # dimensionality of image and word embeddings
hidden_size = 512          # number of features in hidden state of the RNN decoder
num_epochs = 10             # number of training epochs
save_every = 1             # determines frequency of saving model weights
print_every = 100          # determines window for printing average loss
num_layers = 2
lr = 5e-4
opt_name = "adam"
scheduler_name = "cosine_annealing"
training_params = {"opt":opt_name,"scheduler":scheduler_name, "num_layers":num_layers, "lr":lr, "batch_size":batch_size, "vocab_threshold":vocab_threshold, "embed_size":embed_size, "hidden_size":hidden_size, "num_epochs":num_epochs}

# (Optional) TODO #2: Amend the image transform below.
# transform_train = transforms.Compose([ 
#     transforms.ToTensor(),                           # convert the PIL Image to a tensor
#     transforms.Resize(256),                          # smaller edge of image resized to 256
#     transforms.RandomCrop(224),                      # get 224x224 crop from random location
#     transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
#     transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
#                          (0.229, 0.224, 0.225))])

transform_train = get_transform()

# Build data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder. 
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# Define the loss function. 
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# TODO #3: Specify the learnable parameters of the model.
params = list(encoder.parameters())+list(decoder.parameters())

# TODO #4: Define the optimizer.
if opt_name == "adam":
    optimizer = torch.optim.Adam(params,lr)
elif opt_name == "sgd":
    optimizer = torch.optim.SGD(params,lr)
elif opt_name == "rprop":
    optimizer = torch.optim.Rprop(params,lr)

# Set the total number of training steps per epoch.
total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

if scheduler_name == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, total_step/10)
if scheduler_name == "cosine_annealing":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,(total_step)*num_epochs)



mlflow.set_tracking_uri("http://mlflow.cluster.local")
experiment = mlflow.get_experiment_by_name("Image Captioning")
if experiment is None:
    experiment_id = mlflow.create_experiment("Image Captioning")
else:
    experiment_id = experiment.experiment_id
mlflow.start_run(experiment_id=experiment_id)
mlflow.log_params(training_params)
mlflow.log_artifact("./vocab.pkl")


for epoch in range(1, num_epochs+1):
    
    for i_step in range(1, total_step+1):
        
        # Randomly sample a caption length, and sample indices with that length.
        indices = data_loader.dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler
        
        # Obtain the batch.
        images, captions = next(iter(data_loader))

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)
        captions = captions.to(device)
        
        # Zero the gradients.
        decoder.zero_grad()
        encoder.zero_grad()
        
        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs = decoder(features, captions)
        
        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        
        # Backward pass.
        loss.backward()
        
        # Update the parameters in the optimizer.
        optimizer.step()
        scheduler.step()
            
        # Get training statistics.
        stats = {"loss": loss.item(), "perplexity": np.exp(loss.item()), "lr":scheduler.get_last_lr()[0]}

        mlflow.log_metrics(stats, (total_step*(epoch-1))+i_step-1)
            
    # Save the weights.
    if epoch % save_every == 0:

        mlflow.pytorch.log_state_dict(encoder.state_dict(),f"{epoch}/encoder")
        mlflow.pytorch.log_state_dict(decoder.state_dict(),f"{epoch}/decoder")

        torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-%d.pkl' % epoch))
        torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-%d.pkl' % epoch))
    
    

