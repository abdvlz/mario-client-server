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
from pynput.keyboard import Key, Controller
import webbrowser

tf.flags.DEFINE_integer("width", 640, "Screen width")
tf.flags.DEFINE_integer("height", 480, "Screen height")
tf.flags.DEFINE_float("threshold", 0.6, "Threshold for score")
tf.flags.DEFINE_float("alpha", 0.3, "Transparent level")
tf.flags.DEFINE_string("pre_trained_model_path", "mariogame/src/pretrained_model.pb", "Path to pre-trained model")

FLAGS = tf.flags.FLAGS
keyboard = Controller()
webbrowser.open("https://supermarioemulator.com/mario.php")


class Streamer(threading.Thread):

    def __init__(self, hostname, port):
        threading.Thread.__init__(self)

        self.hostname = hostname
        self.port = port
        self.running = False
        self.streaming = False
        self.jpeg = None

    def run(self):
        graph, sess = load_graph(FLAGS.pre_trained_model_path)

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')

        s.bind((self.hostname, self.port))
        print('Socket bind complete')

        payload_size = struct.calcsize("L")

        s.listen(10)
        print('Socket now listening')

        self.running = True

        while self.running:

            print('Start listening for connections...')

            conn, addr = s.accept()
            print("New connection accepted.")

            while True:

                data = conn.recv(payload_size)

                if data:
                    # Read frame size
                    msg_size = struct.unpack("L", data)[0]

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
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    self.jpeg = jpeg
                    self.streaming = True
                    
                    frame = cv2.flip(frame, 1)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes, scores, classes = detect_hands(frame, graph, sess)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    results = predict(boxes, scores, classes, FLAGS.threshold, FLAGS.width, FLAGS.height)
                    
                    print(results)

                    if len(results) == 1:
                        x_min, x_max, y_min, y_max, category = results[0]
                        x = int((x_min + x_max) / 2)
                        y = int((y_min + y_max) / 2)
                        cv2.circle(frame, (x, y), 5, RED, -1)
        
                        if category == "Open" and FLAGS.width / 3 < x <= 2 * FLAGS.width / 3:
                            keyboard.press(Key.right)
                            keyboard.press(Key.space)
                            keyboard.press(Key.space)
                            text = "Jump"
                        elif category == "Closed" and x > 2 * FLAGS.width / 3:
                            keyboard.press(Key.right)
                            text = "Run right"
                        else:
                            keyboard.press(Key.right)
                            text = "Stay"
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
