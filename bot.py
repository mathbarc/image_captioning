import model
import util

import telebot
import torch
import os
import cv2
import string
import numpy
import boto3
import botocore.exceptions
import logging
import io
from util import letterbox_image
import torchvision.transforms.functional


class ImageCaptioningBot:

    def __init__(self):
        token = os.environ["BOT_TOKEN"]
        
        self.s3Client = boto3.client(service_name="s3", endpoint_url=os.environ["BUCKET_ENDPOINT"], aws_access_key_id=os.environ["BUCKET_USER"], aws_secret_access_key=os.environ["BUCKET_SECRET"])
        self.bucket = os.environ["BUCKET_NAME"]
        self.bot = telebot.TeleBot(token, threaded=True)
        self.transform = model.get_inference_transform()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = util.load_vocab("vocab.pkl")

        if torch.cuda.is_available():
            self.model = torch.load("model.pth", map_location=torch.device('cuda'))
        else:
            self.model = torch.load("model.pth", map_location=torch.device('cpu'))

        self.model.eval()
        self.model.to(self.device)


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
            
            img_name = str(message.chat.id)+"/"+str(message.photo[-1].file_id)+"__"+caption+".jpg"

            try:
                bytesIO = io.BytesIO(image_data)
                img_name = img_name.replace(" ","_")
                response = self.s3Client.put_object(Body=bytesIO, Bucket=self.bucket, Key=img_name)
            except botocore.exceptions.ClientError as e:
                logging.error(e)

            
        
        # @self.bot.message_handler(commands=["start"])
        # def start(message:telebot.telebot.types.Message):
        #     self.bot.send_message(chat_id=message.chat.id, text="Send me an image and I will describe it to you!")

        
        self.bot.infinity_polling()

    def inference(self, image):
        
        with torch.no_grad():

            image = letterbox_image(image)
            tensor_image = torchvision.transforms.functional.to_tensor(image)
            tensor_image = tensor_image.unsqueeze(0)

            transformed_image = self.transform(tensor_image)
            output = self.model.sample(transformed_image)

            sentence = ""
            for token in output:
                if token > 1:
                    token_str = self.vocab.idx2word[token]
                    if token_str in string.punctuation:
                        sentence = sentence + token_str
                    else:
                        sentence = sentence + " " + token_str
            return sentence


if __name__=="__main__":
    bot = ImageCaptioningBot()
    bot.run()
