import pickle
import numpy

def load_vocab(vocab_file):
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
        print('Vocabulary successfully loaded from vocab.pkl file!')
        return vocab
    
def letterbox_image(img):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    if img_w > img_h:
        dim = img_w
    else:
        dim = img_h
    
    canvas = numpy.zeros((dim, dim, 3), img.dtype)
    
    x = int((dim-img_w)/2)
    y = int((dim-img_h)/2)
	
    canvas[y:y+img_h, x:x+img_w, :] = img
    
    return canvas