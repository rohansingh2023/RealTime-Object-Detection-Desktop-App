
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import cv2
import tensorflow as tf
import sys

from YOLOv3.yolo_v3 import Yolo_v3
from YOLOv3.utils import load_images, load_class_names, draw_boxes, draw_frame

tf.compat.v1.disable_eager_execution()

_MODEL_SIZE = (416, 416)
_CLASS_NAMES_FILE = './YOLOv3/data/labels/coco.names'
_MAX_OUTPUT_SIZE = 20

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def run(self):
        self.ThreadActive = True

        class_names = load_class_names(_CLASS_NAMES_FILE)
        n_classes = len(class_names)

        iou_threshold = 0.5
        confidence_threshold = 0.5

        model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                        max_output_size=_MAX_OUTPUT_SIZE,
                        iou_threshold=iou_threshold,
                        confidence_threshold=confidence_threshold)

        inputs = tf.compat.v1.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(scope='yolo_v3_model'))

        with tf.compat.v1.Session() as sess:
            saver.restore(sess, './YOLOv3/weights/model.ckpt')

            cap = cv2.VideoCapture(0)
            frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                            cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter('.YOLOv3/detections/detections.mp4', fourcc, fps,
                                    (int(frame_size[0]), int(frame_size[1])))

            while self.ThreadActive:
                ret, frame = cap.read()
                if ret:
                    resized_frame = cv2.resize(frame, dsize=_MODEL_SIZE[::-1],
                                                interpolation=cv2.INTER_NEAREST)
                    detection_result = sess.run(detections,
                                                feed_dict={inputs: [resized_frame]})

                    draw_frame(frame, frame_size, detection_result,
                                class_names, _MODEL_SIZE)
                    
                    Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    FlippedImage = cv2.flip(Image, 1)
                    ConvertToQtFormat = QImage(FlippedImage.data, 
                    FlippedImage.shape[1], FlippedImage.shape[0],
                    QImage.Format_RGB888)
                    Pic = ConvertToQtFormat.scaled(640,480, 
                    Qt.KeepAspectRatio)
                    self.ImageUpdate.emit(Pic)

                    out.write(frame)

    def stop(self):
        self.ThreadActive = False
        self.quit()
