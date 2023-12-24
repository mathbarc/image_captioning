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
from model import ImageCaptioner, get_transform, get_inference_transform
import mlflow


## TODO #1: Select appropriate values for the Python variables below.
batch_group_size = 1   # batch group size
batch_size = 8          # batch size
vocab_threshold = 20        # minimum word count threshold
vocab_from_file = True    # if True, load existing vocab file
embed_size = 512           # dimensionality of image and word embeddings
hidden_size = 256         # number of features in hidden state of the RNN decoder
num_epochs = 10             # number of training epochs
save_every = 1000             # determines frequency of saving model weights
num_layers = 1
lr = 1e-3
last_every = 100
opt_name = "adam"
scheduler_name = "cosine_annealing"
dropout = 0.3
grad_clip = 1

transform_train = get_transform()

# Build data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file,
                         num_workers=8)

transform_valid = get_inference_transform()
data_loader_valid = get_loader(transform=transform_valid,
                         mode='valid',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=True,
                         num_workers=8)


# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)
training_params = {"opt":opt_name,
                   "scheduler":scheduler_name, 
                   "num_layers":num_layers, 
                   "lr":lr, 
                   "batch_size":batch_size, 
                   "vocab_threshold":vocab_threshold, 
                   "embed_size":embed_size, 
                   "hidden_size":hidden_size, 
                   "dropout": dropout,
                   "num_epochs":num_epochs, 
                   "batch_group_size":batch_group_size, 
                   "vocab_size":vocab_size,
                   "grad_clip":grad_clip}

# Initialize the encoder and decoder. 
model = ImageCaptioner(embed_size, hidden_size, vocab_size, num_layers, dropout=dropout, pretreined=True)

# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the trained weights.
# encoder.load_state_dict(torch.load(os.path.join('./models/encoder-4.pkl'),map_location=device))
# decoder.load_state_dict(torch.load(os.path.join('./models/decoder-4.pkl'),map_location=device))

# Define the loss function. 
criterion = nn.CrossEntropyLoss(reduction="mean").cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss(reduction="mean")

# TODO #3: Specify the learnable parameters of the model.
params = list(model.encoder.parameters())+list(model.decoder.parameters())


# Set the total number of training steps per epoch.
total_step = 20000
# total_step = 2000

mlflow.set_tracking_uri("http://mlflow.cluster.local")
experiment = mlflow.get_experiment_by_name("Image Captioning")
if experiment is None:
    experiment_id = mlflow.create_experiment("Image Captioning")
else:
    experiment_id = experiment.experiment_id
mlflow.start_run(experiment_id=experiment_id)
mlflow.log_params(training_params)
mlflow.log_artifact("./vocab.pkl")


if opt_name == "adam":
        optimizer = torch.optim.Adam(params,lr)
elif opt_name == "sgd":
    optimizer = torch.optim.SGD(params,lr)
elif opt_name == "rprop":
    optimizer = torch.optim.Rprop(params,lr)

if scheduler_name == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, total_step/20, 0.1)
if scheduler_name == "cosine_annealing":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_step)

acc_loss = 0

hidden = model.decoder.init_hidden(batch_size)

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
        model.zero_grad()
    
    # Pass the inputs through the CNN-RNN model.
    output, hidden = model(images, captions, hidden)
    hidden = model.decoder.repackage_hidden(hidden)
    
    # Calculate the batch loss.
    loss = criterion(output.view(-1, vocab_size), captions.view(-1))
    acc_loss += loss.item()

    # Backward pass.
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    
        
    # Get training statistics.
    if (i_step-1)%batch_group_size == batch_group_size-1:
        # Update the parameters in the optimizer.
        optimizer.step()
        scheduler.step()
        
        
    if int(((i_step)/batch_group_size)%batch_group_size)==batch_group_size-1:
        acc_loss = acc_loss/batch_group_size
        stats = {"loss": acc_loss, "lr":scheduler.get_last_lr()[0]}
        mlflow.log_metrics(stats, i_step)
    
    if (i_step-1)%batch_group_size == batch_group_size-1:
        acc_loss = 0

    if int(((i_step)/batch_group_size)%last_every)==last_every-1:
        mlflow.pytorch.log_model(model,f"image_captioner_last",extra_files=["model.py"])


    
            
    # Save the weights.
    if (i_step-1)%save_every == save_every - 1:

        hidden_val = model.decoder.init_hidden(batch_size)

        mlflow.pytorch.log_model(model,f"{i_step}/image_captioner",extra_files=["model.py"])
        # torch.save(model, os.path.join('./models', 'image_captioner-%d.pkl' % epoch))

        model.eval()

        acc_test_loss = 0
        count = 0
        for i in tqdm.tqdm(range(0,100)):

            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader_valid.dataset.get_train_indices()
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            data_loader_valid.batch_sampler.sampler = new_sampler

            images, captions = next(iter(data_loader_valid))

            model.zero_grad()

            images = images.to(device)
            captions = captions.to(device)

            output, hidden_val = model(images, captions, hidden_val)
            hidden_val = model.decoder.repackage_hidden(hidden_val)
            
            # Calculate the batch loss.
            loss = criterion(output.view(-1, vocab_size), captions.view(-1))
            acc_test_loss += loss.item()
            count+=1
        
        acc_test_loss = acc_test_loss/count
        stats = {"loss_valid": acc_test_loss}
        mlflow.log_metrics(stats, i_step)

        model.train()
            
            

            
    
    

