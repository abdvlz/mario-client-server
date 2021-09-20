FROM tiangolo/uwsgi-nginx:python3.8

EXPOSE 5000 8080

WORKDIR /app

COPY . .

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y python3-opencv

RUN pip3 install -r requirements.txt

CMD ["python3", "server.py"]