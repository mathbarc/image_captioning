import cv2
import time
import numpy
import pickle



def infer_complete(blob):
    model = cv2.dnn.readNetFromONNX("mlruns/613000375512790258/31f32b8c88b84ad7acb1764ca1acb32b/artifacts/last.onnx")
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    with open("mlruns/613000375512790258/31f32b8c88b84ad7acb1764ca1acb32b/artifacts/simple_vocab.pkl","rb") as file:
        dictionary = pickle.load(file)
    
    
    start = time.time()
    
    model.setInput(blob, "images")
    tokens = model.forward("output")
    
    end = time.time()
    
    print(end-start)
    print(tokens)
    print([[dictionary["idx2word"][token] for token in response ]for response in tokens])
    


def infer_subcomponent(blob):
    encoder = cv2.dnn.readNetFromONNX("test_encoder.onnx")
    encoder.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    encoder.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    embed = cv2.dnn.readNetFromONNX("test_embed.onnx")
    embed.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    embed.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    decoder = cv2.dnn.readNetFromONNX("test_decoder.onnx")
    decoder.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    decoder.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    start = time.time()


    c = numpy.zeros((1, 1, 128))
    h = numpy.zeros((1, 1, 128))

    tokens = []


    encoder.setInput(blob,"images")
    img_features = encoder.forward("output").reshape(1,1,256)


    decoder.setInput(img_features, "embed")
    decoder.setInput(h, "state_input_h")
    decoder.setInput(c, "state_input_c")

    token, h, c = decoder.forward(("output", "state_output_h", "state_output_c"))


    end = time.time()

    print(end-start)


if __name__=="__main__":
    
    img = cv2.imread("/data/hd1/Dataset/Coco/test2017/000000000001.jpg")

    blob = cv2.dnn.blobFromImage(img, 1/255, (480,480),swapRB=True)
    
    
    
    
    
    infer_complete(blob)