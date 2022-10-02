FROM python:3.10

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y cmake

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install -r requirements.txt

COPY . .

WORKDIR /code

CMD [ "python3", "create_thumbnail.py","../data/videos" ]
