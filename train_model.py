import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import tqdm

from data_loader import get_loader
from model import create_encoder, DecoderRNN, get_transform, get_inference_transform
import mlflow


## TODO #1: Select appropriate values for the Python variables below.
batch_group_size = 4   # batch group size
batch_size = 64          # batch size
vocab_threshold = 20        # minimum word count threshold
vocab_from_file = False    # if True, load existing vocab file
embed_size = 64           # dimensionality of image and word embeddings
hidden_size = 128          # number of features in hidden state of the RNN decoder
num_epochs = 4             # number of training epochs
save_every = 1             # determines frequency of saving model weights
num_layers = 2
lr = 1e-3
print_every = 10
opt_name = "adam"
scheduler_name = "cosine_annealing"


transform_train = get_transform()

# Build data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)

transform_valid = get_inference_transform()
data_loader_valid = get_loader(transform=transform_valid,
                         mode='valid',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=True)


# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)
training_params = {"opt":opt_name,"scheduler":scheduler_name, "num_layers":num_layers, "lr":lr, "batch_size":batch_size, "vocab_threshold":vocab_threshold, "embed_size":embed_size, "hidden_size":hidden_size, "num_epochs":num_epochs, "batch_group_size":batch_group_size, "vocab_size":vocab_size}

# Initialize the encoder and decoder. 
encoder = create_encoder(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# Load the trained weights.
# encoder.load_state_dict(torch.load(os.path.join('./models/encoder-4.pkl'),map_location=device))
# decoder.load_state_dict(torch.load(os.path.join('./models/decoder-4.pkl'),map_location=device))

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

acc_loss = 0
for epoch in range(1, num_epochs+1):
    
    for i_step in tqdm.tqdm(range(1, total_step+1)):

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

        if (i_step-1)%batch_group_size == 0:
            # Zero the gradients.
            decoder.zero_grad()
            encoder.zero_grad()
        
        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs = decoder(features, captions)
        
        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        acc_loss += loss.item()

        # Backward pass.
        loss.backward()
        
            
        # Get training statistics.
        if (i_step-1)%batch_group_size == batch_group_size-1:
            
            # Update the parameters in the optimizer.
            optimizer.step()
            scheduler.step()

            if int(((i_step)/batch_group_size)%print_every)==0:
                acc_loss = acc_loss/batch_group_size
                stats = {"loss": acc_loss, "perplexity": np.exp(acc_loss), "lr":scheduler.get_last_lr()[0]}
                mlflow.log_metrics(stats, (total_step*(epoch-1))+i_step-1)
            
            acc_loss = 0

            
            
            
    # Save the weights.
    if epoch % save_every == 0:

        mlflow.pytorch.log_model(encoder,f"{epoch}/encoder")
        mlflow.pytorch.log_model(decoder,f"{epoch}/decoder",extra_files="model.py")

        torch.save(decoder, os.path.join('./models', 'decoder-%d.pkl' % epoch))
        torch.save(encoder, os.path.join('./models', 'encoder-%d.pkl' % epoch))

        encoder.eval()
        decoder.eval()


        acc_test_loss = 0
        count = 0
        for i in tqdm.tqdm(range(0,100)):

            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader_valid.dataset.get_train_indices()
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            data_loader_valid.batch_sampler.sampler = new_sampler

            images, captions = next(iter(data_loader_valid))

            decoder.zero_grad()
            encoder.zero_grad()

            images = images.to(device)
            captions = captions.to(device)

            features = encoder(images)
            outputs = decoder(features, captions)
            
            # Calculate the batch loss.
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            acc_test_loss += loss.item()
            count+=1
        
        acc_test_loss = acc_test_loss/count
        stats = {"loss_valid": acc_test_loss, "perplexity_valid": np.exp(acc_test_loss)}
        mlflow.log_metrics(stats, (total_step*(epoch)))

        encoder.train()
        decoder.train()
            
            

            
    
    

