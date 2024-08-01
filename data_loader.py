import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from torchvision import io
from torchvision.transforms.functional import to_tensor
import cv2
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json
import config
from util import letterbox_image


def get_loader(mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc=config.DATA_DIR, 
               download_directly=False):
    """Returns the data loader.
    Args:
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary. 
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading 
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """
    if not os.path.exists("tokenizers"):
        nltk.download('punkt',"./")
    
    assert mode in ['train', 'test', 'valid'], "mode must be one of 'train' or 'test'."
    if vocab_from_file==False: assert mode=='train', "To generate vocab from captions file, must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    if mode == 'train':
        if vocab_from_file==True: assert os.path.exists(vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(cocoapi_loc, 'train2017/')
        annotations_file = os.path.join(cocoapi_loc, 'annotations/captions_train2017.json')
    if mode == 'test':
        assert batch_size==1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file==True, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, 'test2014/')
        annotations_file = os.path.join(cocoapi_loc, 'annotations/image_info_test2014.json')

    if mode == 'valid':
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file==True, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, 'val2017/')
        annotations_file = os.path.join(cocoapi_loc, 'annotations/captions_val2017.json')

    # COCO caption dataset.
    dataset = CoCoDataset(mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder,
                          download_directly=download_directly)

    if mode == 'train':
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader

class CoCoDataset(data.Dataset):
    
    def __init__(self, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder, download_directly):
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        
        self.img_folder = img_folder
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)
        self.download_directly = download_directly
        if self.mode == 'train' or self.mode == "valid":
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item['file_name'] for item in test_info['images']]
            # self.coco = COCO(annotations_file)
            # self.ids = list(self.coco.anns.keys())
        
    def __getitem__(self, index):
        # obtain image and caption if in training mode
        if self.mode == 'train' or self.mode == "valid":
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']


            img_name = self.coco.loadImgs(img_id)[0]['file_name']
            path = os.path.join(self.img_folder, img_name)
            if not os.path.exists(path):
                self.coco.download(self.img_folder, [img_id])

            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image = letterbox_image(image)
            image=cv2.resize(image, (640,640))

            image = to_tensor(image)
            

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # return pre-processed image and caption tensors
            return image, caption
        


        # obtain image if in test mode
        else:
            path = self.paths[index]
            original_image = cv2.imread(os.path.join(self.img_folder, path))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            image = letterbox_image(original_image)
            image = cv2.resize(image, (640,640))

            image = to_tensor(image)

            # return original image and pre-processed image tensor
            return original_image, image

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def encode_caption(self, caption):
        sample_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption_tokens = []

        caption_tokens.append(self.vocab(self.vocab.start_word))
        caption_tokens.extend([self.vocab(token) for token in sample_tokens])
        caption_tokens.append(self.vocab(self.vocab.end_word))
        return caption_tokens

    def __len__(self):
        if self.mode == 'train' or self.mode == "valid":
            return len(self.ids)
        else:
            return len(self.paths)

if __name__=="__main__":

    from torchvision import transforms

    transform_train = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Resize(480, antialias=True),                          # smaller edge of image resized to 256
    transforms.RandomCrop(416),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    #transforms.Normalize((0.5, 0.5, 0.5),      # normalize image for pre-trained model
    #                     (0.5, 0.5, 0.5))
    ])

    # Set the minimum word count threshold.
    vocab_threshold = 5

    # Specify the batch size.
    batch_size = 10

    data_loader = get_loader(
                         mode='train',
                         batch_size=batch_size,
                         vocab_from_file=True)

    images, captions = next(iter(data_loader))
    
    print('images.shape:', images.shape)
    print('captions.shape:', captions.shape)
