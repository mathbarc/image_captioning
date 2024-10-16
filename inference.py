import cv2
import time
import numpy
import pickle
from typing import Tuple


def load_model(model_path:str = "best.onnx") -> Tuple[cv2.dnn.Net, float, float, float]:
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    gflops = 0

    input_size = (1,3,480,480)
    gflops = model.getFLOPS(input_size)
    memory_weights, memory_blobs = model.getMemoryConsumption(input_size)
    
    return model, gflops*1e-9, memory_weights*1e-6, memory_blobs*1e-6

def load_vocab(vocab_path = "simple_vocab.pkl"):
    with open(vocab_path,"rb") as file:
        vocab = pickle.load(file)
    return vocab

def infer_model(blob, model):
    
    start = time.time()
    
    model.setInput(blob, "images")
    tokens = model.forward("output")
    
    end = time.time()
    
    return tokens, end-start

def translate_tokens(tokens, vocab):
    tokens_str = [[vocab["idx2word"][token] for token in response ]for response in tokens]
    caption = []
    
    for img_tokens in tokens_str:
        for token in img_tokens:
            if token == "<end>":
                caption[-1] = caption[-1].lstrip(" ")
                break
            elif token == "<start>":
                caption.append("")
            elif token in [".",","]:
                caption[-1] += token
            else:
                caption[-1] += " "+token
    
    return caption
    
    

if __name__=="__main__":
    
    model, gflops, memory_weights, memory_blob = load_model("best.onnx")
    
    print(gflops, "GFLOPs")   
    print(memory_weights, "MB") 
    print(memory_blob, "MB") 
    print(memory_weights+memory_blob, "MB") 
    
    img = cv2.imread("/data/ssd1/Datasets/Coco/test2017/000000004366.jpg")
    blob = cv2.dnn.blobFromImage(img, 1/255, (480,480),swapRB=True)
    
    tokens, infer_time = infer_model(blob, model)
    print(tokens)
    
    vocab = load_vocab("simple_vocab.pkl")
    translated_tokens = translate_tokens(tokens, vocab)
    
    print(translated_tokens)
    print(infer_time, 's')
    
    
    
