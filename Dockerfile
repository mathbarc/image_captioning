FROM python:3.8.2-slim

RUN apt update;DEBIAN_FRONTEND=noninteractive apt install libavcodec-dev libavformat-dev libavresample-dev libswscale-dev -y; apt clean --dry-run; apt autoclean;
RUN pip install torch==1.12.0 torchvision==0.13.0 pyTelegramBotAPI numpy opencv-python==4.6.0.66 boto3

COPY simple_vocab.pkl ./
COPY models/best/encoder.pth ./models/encoder.pth
COPY models/best/decoder.pth ./models/decoder.pth

COPY model.py model.py
COPY util.py util.py
COPY bot.py bot.py

ENTRYPOINT [ "python3", "bot.py" ]
