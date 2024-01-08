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
batch_size = 80          # batch size
vocab_threshold = 20        # minimum word count threshold
vocab_from_file = False    # if True, load existing vocab file
num_epochs = 10             # number of training epochs
save_every = 1000             # determines frequency of saving model weights
num_layers = 2
lr = 1e-4
last_every = 100
opt_name = "adam"
scheduler_name = "step"
dropout = 0.1

transform_train = get_transform()

# Build data loader.
data_loader = get_loader(mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file,
                         num_workers=8)

transform_valid = get_inference_transform()
data_loader_valid = get_loader(mode='valid',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=True,
                         num_workers=8)


# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)
embed_size = 256           # dimensionality of image and word embeddings
hidden_size = 512         # number of features in hidden state of the RNN decoder
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
                   "vocab_size":vocab_size}

# Initialize the encoder and decoder. 
model = ImageCaptioner(embed_size, hidden_size, vocab_size, num_layers, dropout=dropout, pretreined=False)
model.train()

# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the trained weights.
# encoder.load_state_dict(torch.load(os.path.join('./models/encoder-4.pkl'),map_location=device))
# decoder.load_state_dict(torch.load(os.path.join('./models/decoder-4.pkl'),map_location=device))

# Define the loss function. 
criterion = nn.CrossEntropyLoss(reduction="mean").cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss(reduction="mean")

# TODO #3: Specify the learnable parameters of the model.
# params = list(model.encoder.parameters())+list(model.decoder.parameters())
params = model.parameters()


# Set the total number of training steps per epoch.
total_step = 2000
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(total_step/2), 0.1)
elif scheduler_name == "cosine_annealing":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_step)
elif scheduler_name == "constant":
    scheduler = None

acc_loss = 0

hidden = model.decoder.init_hidden(batch_size)
best_loss = 100

for i_step in tqdm.tqdm(range(1, total_step+1)):

    # Randomly sample a caption length, and sample indices with that length.
    indices = data_loader.dataset.get_train_indices()
    # Create and assign a batch sampler to retrieve a batch with the sampled indices.
    new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    data_loader.batch_sampler.sampler = new_sampler
    
    # Obtain the batch.
    images, captions = next(iter(data_loader))
    images = transform_train(images)

    # Move batch of images and captions to GPU if CUDA is available.
    images = images.to(device)
    captions = captions.to(device)

    
    # Zero the gradients.
    optimizer.zero_grad()
    
    # Pass the inputs through the CNN-RNN model.
    output, hidden = model(images, captions, hidden)
    hidden = model.decoder.repackage_hidden(hidden)
    
    # Calculate the batch loss.
    loss = criterion(output.view(-1, vocab_size), captions.view(-1))
    acc_loss = loss.item()

    # Backward pass.
    loss.backward()

    torch.nn.utils.clip_grad_norm_(params, 1)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
        
    if scheduler is not None:
        current_lr = scheduler.get_last_lr()[0]
    else:
        current_lr = lr
    stats = {"loss": acc_loss, "lr":current_lr}
    mlflow.log_metrics(stats, i_step)
    
    
    if int(i_step%last_every)==last_every-1:
        mlflow.pytorch.log_model(model,"last",extra_files=["model.py"])

    
    # Save the weights.
    # if (i_step-1)%save_every == save_every - 1:

    #     hidden_val = model.decoder.init_hidden(batch_size)
    #     # torch.save(model, os.path.join('./models', 'image_captioner-%d.pkl' % epoch))

    #     model.eval()

    #     acc_test_loss = 0
    #     count = 0
    #     for i in tqdm.tqdm(range(0,100)):

    #         # Randomly sample a caption length, and sample indices with that length.
    #         indices = data_loader_valid.dataset.get_train_indices()
    #         # Create and assign a batch sampler to retrieve a batch with the sampled indices.
    #         new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    #         data_loader_valid.batch_sampler.sampler = new_sampler

    #         images, captions = next(iter(data_loader_valid))
    #         images = transform_valid(images)

    #         images = images.to(device)
    #         captions = captions.to(device)

    #         output, hidden_val = model(images, captions, hidden_val)
    #         hidden_val = model.decoder.repackage_hidden(hidden_val)
            
    #         # Calculate the batch loss.
    #         loss = criterion(output.view(-1, vocab_size), captions.view(-1))
    #         acc_test_loss += loss.item()
    #         count+=1
        
    #     acc_test_loss = acc_test_loss/count
    #     stats = {"loss_valid": acc_test_loss}
    #     mlflow.log_metrics(stats, i_step)

    #     if best_loss > acc_test_loss:
    #         best_loss = acc_test_loss
    #         mlflow.pytorch.log_model(model,"best",extra_files=["model.py"])

    #     model.train()
            
            

            
    
    

