FROM python:3.8.2-slim

ARG BOT_TOKEN
ENV BOT_TOKEN=${BOT_TOKEN}

RUN apt update;DEBIAN_FRONTEND=noninteractive apt install libavcodec-dev libavformat-dev libavresample-dev libswscale-dev -y; apt clean --dry-run; apt autoclean;
RUN pip install torch==2.0.1 torchvision==0.15.2 numpy opencv-python==4.6.0.66 
RUN pip install pyTelegramBotAPI boto3 nltk pycocotools

WORKDIR /home/bot

COPY vocab.pkl ./vocab.pkl
COPY model.pth ./model.pth

COPY model.py model.py
COPY vocabulary.py vocabulary.py
COPY util.py util.py
COPY bot.py bot.py

ENTRYPOINT [ "python3", "bot.py" ]