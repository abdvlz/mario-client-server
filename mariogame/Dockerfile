FROM ubuntu

WORKDIR /app

COPY . /app

RUN apt-get update \
&& apt-get install -y python3-pip python3-dev \
&& cd /usr/local/bin \
&& pip3 install --upgrade pip

ARG DEBIAN_FRONTEND=noninteractive
  
RUN apt update
RUN apt install -y libgl1-mesa-glx \
&& apt-get install -y libgtk2.0-dev

RUN pip install -r requirements.txt

CMD ["python3", "./mario.py"]