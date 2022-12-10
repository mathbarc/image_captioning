import model
import util

import telebot
import torch
import os
import cv2
import string
import numpy

class ImageCaptioningBot:

    def __init__(self):
        token = os.environ["BOT_TOKEN"]
        self.bot = telebot.TeleBot(token, threaded=True)
        self.transform = model.get_inference_transform()

        embed_size = 256
        hidden_size = 512


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = util.load_vocab("simple_vocab.pkl")

        self.encoder = model.EncoderCNN(embed_size)
        self.encoder.eval()
        self.decoder = model.DecoderRNN(embed_size, hidden_size, len(self.vocab["word2idx"]))
        self.decoder.eval()

        # Load the trained weights.
        self.encoder.load_state_dict(torch.load(os.path.join('./models/encoder.pkl'),map_location=self.device))
        self.decoder.load_state_dict(torch.load(os.path.join('./models/decoder.pkl'),map_location=self.device))

        self.encoder.to(self.device)
        self.decoder.to(self.device)


    def run(self):
        @self.bot.message_handler(content_types=["photo"])
        def describe(message:telebot.telebot.types.Message):
            image_path = self.bot.get_file(message.photo[-1].file_id)
            image_data = self.bot.download_file(image_path.file_path)
            data = numpy.fromstring(image_data,dtype=numpy.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            caption = self.inference(image)
            self.bot.send_message(chat_id=message.chat.id, text=caption)
        
        # @self.bot.message_handler(commands=["start"])
        # def start(message:telebot.telebot.types.Message):
        #     self.bot.send_message(chat_id=message.chat.id, text="Send me an image and I will describe it to you!")

        
        self.bot.infinity_polling()

    def inference(self, image):
        
        size = image.shape

        if size[0] < size[1]:
            diff = int((size[1]-size[0])/2)
            image = image[0:size[0], diff:diff+size[0]]
        else:
            diff = int((size[0]-size[1])/2)
            image = image[diff:diff+size[1], 0:size[1]]
            

        transformed_image = self.transform(image)
        transformed_image = transformed_image.to(self.device)
        size = transformed_image.size()
        reshaped_image = torch.reshape(transformed_image, [1,size[0],size[1],size[2]])

        # Obtain the embedded image features.
        features = self.encoder(reshaped_image).unsqueeze(1)

        # Pass the embedded image features through the model to get a predicted caption.
        output = self.decoder.sample(features)

        sentence = ""
        for token in output:
            if token > 1:
                token_str = self.vocab["idx2word"][token]
                if token_str in string.punctuation:
                    sentence = sentence + token_str
                else:
                    sentence = sentence + " " + token_str
        return sentence


if __name__=="__main__":
    bot = ImageCaptioningBot()
    bot.run()
