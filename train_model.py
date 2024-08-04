import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from torchmetrics import Accuracy
import tqdm

from data_loader import get_loader
from model import ImageCaptioner, get_transform, get_inference_transform
import mlflow

import schedulers


## TODO #1: Select appropriate values for the Python variables below.
batch_size = 8          # batch size
vocab_threshold = 20        # minimum word count threshold
vocab_from_file = False    # if True, load existing vocab file

lr = 1e-3
last_every = 100
opt_name = "adam"
scheduler_name = "logistic"
dropout = 0.4
backbone_type = "efficientnet_v2"

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


save_every = 500

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)
embed_size = 1024          # dimensionality of image and word embeddings
hidden_size = 256         # number of features in hidden state of the RNN decoder
num_layers = 1
total_step = 100000
training_params = {"opt":opt_name,
                   "scheduler":scheduler_name, 
                   "num_layers":num_layers, 
                   "lr":lr, 
                   "batch_size":batch_size, 
                   "vocab_threshold":vocab_threshold, 
                   "embed_size":embed_size, 
                   "hidden_size":hidden_size, 
                   "dropout": dropout,
                   "steps":total_step, 
                   "vocab_size":vocab_size,
                   "backbone_type":backbone_type}

# Initialize the encoder and decoder. 
model = ImageCaptioner(backbone_type,embed_size, hidden_size, vocab_size, num_layers, dropout=dropout, max_len=max(data_loader.dataset.caption_lengths))
model.train()

# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the trained weights.
# encoder.load_state_dict(torch.load(os.path.join('./models/encoder-4.pkl'),map_location=device))
# decoder.load_state_dict(torch.load(os.path.join('./models/decoder-4.pkl'),map_location=device))

# Define the loss function. 
criterion = nn.CrossEntropyLoss()
eval_criterion = Accuracy(task="multiclass", num_classes=vocab_size)

if torch.cuda.is_available():
    criterion = criterion.cuda()
    eval_criterion = eval_criterion.cuda()

# TODO #3: Specify the learnable parameters of the model.
# params = list(model.encoder.parameters())+list(model.decoder.parameters())
params = model.parameters()


# Set the total number of training steps per epoch.

# total_step = 2000

# mlflow.set_tracking_uri("http://mlflow.cluster.local")
experiment = mlflow.get_experiment_by_name("Image Captioning")
if experiment is None:
    experiment_id = mlflow.create_experiment("Image Captioning")
else:
    experiment_id = experiment.experiment_id
mlflow.start_run(experiment_id=experiment_id)
mlflow.log_params(training_params)
mlflow.log_artifact("simple_vocab.pkl")


if opt_name == "adam":
        optimizer = torch.optim.Adam(params,lr)
elif opt_name == "sgd":
    optimizer = torch.optim.SGD(params,lr)
elif opt_name == "asgd":
    optimizer = torch.optim.ASGD(params,lr)
elif opt_name == "rprop":
    optimizer = torch.optim.Rprop(params,lr)

if scheduler_name == "logistic":
    scheduler = schedulers.RampUpLogisticDecayScheduler(optimizer, lr, 1e-5, total_step, 1000)
if scheduler_name == "cosine":
    scheduler = schedulers.RampUpCosineDecayScheduler(optimizer, lr, 1e-5, total_step, 1000)
elif scheduler_name == "cosine_annealing":
    scheduler = schedulers.RampUpCosineAnnealingScheduler(optimizer, lr, 1e-5, 1000, 1000,2)
elif scheduler_name == "constant":
    scheduler = schedulers.RampUpScheduler(optimizer, lr, 1000)

acc_loss = 0
best_loss = 0

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
    output = model.compute_gradients(images, captions)
    
    # Calculate the batch loss.
    loss = criterion(output.view(-1, vocab_size), captions.view(-1))
    acc_loss = loss.item()

    # Backward pass.
    loss.backward()

    #torch.nn.utils.clip_grad_norm_(params, 4)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
        current_lr = scheduler.get_last_lr()
    else:
        current_lr = lr

    stats = {"loss": acc_loss, "lr":current_lr}

    try:
        mlflow.log_metrics(stats, i_step)
    except mlflow.MlflowException as e:
        print(e.message)
    
    if int(i_step%last_every)==last_every-1:
        continue_uploading = True
        while continue_uploading:
            try:
                # mlflow.pytorch.log_model(model,"last",extra_files=["model.py"])
                model.save("last")
                mlflow.log_artifact("last.onnx")
                continue_uploading = False
            except mlflow.MlflowException as e:
                print(e.message)
                time.sleep(5)
    

    # Save the weights.
    if (i_step-1)%save_every == save_every - 1:

        model.eval()

        acc_test_loss = 0
        count = 0
        
        with torch.no_grad():

            for images, captions in tqdm.tqdm(data_loader_valid.dataset):

                images = transform_valid(images)
                images = images.unsqueeze(0)
                captions = captions.unsqueeze(0)

                images = images.to(device)
                captions = captions.to(device)

                output = model(images)
                
                if output.shape[1] < captions.shape[1]:
                    captions = captions[:,:output.shape[1]]
                elif output.shape[1] > captions.shape[1]:
                    output = output[:,:captions.shape[1]]
                
                # Calculate the batch loss.
                acc = eval_criterion(output, captions)
                acc_test_loss += acc.item()
                count+=1
            
        acc_test_loss = acc_test_loss/count
        stats = {"loss_valid": acc_test_loss}
        mlflow.log_metrics(stats, i_step)

        if best_loss < acc_test_loss:
            best_loss = acc_test_loss
            
            continue_uploading = True
            while continue_uploading:
                try:
                    # mlflow.pytorch.log_model(model,"best",extra_files=["model.py"])
                    model.save("best")
                    mlflow.log_artifact("best.onnx")
                    continue_uploading = False
                except mlflow.MlflowException as e:
                    print(e.message)
                    time.sleep(5)
        

        model.train()
            
            
# mlflow.pytorch.log_model(model,"final",extra_files=["model.py"])
model.save("final")
mlflow.log_artifact("final.onnx")
            
    
    

