"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import tensorflow.compat.v1 as tf
import cv2
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

def main(frame, graph, sess):
    cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, scores, classes = detect_hands(frame, graph, sess)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = predict(boxes, scores, classes, FLAGS.threshold, FLAGS.width, FLAGS.height)

    if len(results) == 1:
        x_min, x_max, y_min, y_max, category = results[0]
        x = int((x_min + x_max) / 2)
        y = int((y_min + y_max) / 2)
        cv2.circle(frame, (x, y), 5, RED, -1)

        if category == "Open" and FLAGS.width / 3 < x <= 2 * FLAGS.width / 3:
            keyboard.press(Key.space)
            text = "Jump"
        else:
            keyboard.press(Key.right)
            text = "Stay"

        cv2.putText(frame, "{}".format(text), (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (int(FLAGS.width / 3), FLAGS.height), ORANGE, -1)
    cv2.rectangle(overlay, (int(2 * FLAGS.width / 3), 0), (FLAGS.width, FLAGS.height), ORANGE, -1)
    cv2.addWeighted(overlay, FLAGS.alpha, frame, 1 - FLAGS.alpha, 0, frame)
    cv2.imshow('Detection', frame)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
