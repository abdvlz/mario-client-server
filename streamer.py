import cv2
import numpy
import socket
import struct
import threading
import time
from io import BytesIO
import tensorflow.compat.v1 as tf
import multiprocessing as _mp
from mariogame.src.utils import load_graph, mario, detect_hands, predict
from mariogame.src.config import ORANGE, RED, GREEN
import os

tf.flags.DEFINE_integer("width", 640, "Screen width")
tf.flags.DEFINE_integer("height", 480, "Screen height")
tf.flags.DEFINE_float("threshold", 0.6, "Threshold for score")
tf.flags.DEFINE_float("alpha", 0.3, "Transparent level")
tf.flags.DEFINE_string("pre_trained_model_path", "mariogame/src/pretrained_model.pb", "Path to pre-trained model")

FLAGS = tf.flags.FLAGS

class Streamer(threading.Thread):

    def __init__(self, hostname, port):
        threading.Thread.__init__(self)

        self.hostname = hostname
        self.port = port
        self.running = False
        self.streaming = False
        self.jpeg = None
        self.gesture = ""

    def run(self):
        graph, sess = load_graph(FLAGS.pre_trained_model_path)

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')

        s.bind((self.hostname, self.port))
        print('Socket bind complete')

        payload_size = struct.calcsize("<L")

        s.listen(10)
        print('Socket now listening')

        self.running = True

        while self.running:

            print('Start listening for connections...')

            conn, addr = s.accept()
            print("New connection accepted.")

            counter=0
            target=5

            while True:

                data = conn.recv(payload_size)

                if data:
                    # Read frame size
                    msg_size = struct.unpack("<L", data)[0]

                    # Read the payload (the actual frame)
                    data = b''
                    while len(data) < msg_size:
                        missing_data = conn.recv(msg_size - len(data))
                        if missing_data:
                            data += missing_data
                        else:
                            # Connection interrupted
                            self.streaming = False
                            break

                    # Skip building frame since streaming ended
                    if self.jpeg is not None and not self.streaming:
                        continue

                    # Convert the byte array to a 'jpeg' format
                    memfile = BytesIO()
                    memfile.write(data)
                    memfile.seek(0)

                    frame = numpy.load(memfile)
                    print(frame)

                    ret, jpeg = cv2.imencode('.jpg', frame)
                    self.jpeg = jpeg
                    self.streaming = True

                    if counter == target:
                        if ret:
                            counter = 0
                            #frame = cv2.flip(frame, 1)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            boxes, scores, classes = detect_hands(frame, graph, sess)
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            results = predict(boxes, scores, classes, FLAGS.threshold, FLAGS.width, FLAGS.height)
                            if len(results) == 1:
                                self.gesture = results
                            else:
                                self.gesture = ""
                    else:
                        counter += 1 

                else:
                    conn.close()
                    print('Closing connection...')
                    self.streaming = False
                    self.running = False
                    self.jpeg = None
                    break

        print('Exit thread.')

    def stop(self):
        self.running = False

    def get_jpeg(self):
        return self.jpeg.tobytes()

    def get_gesture(self):
        return self.gesture
