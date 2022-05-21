# coding: utf-8
# Object Detection Demo
import argparse
import cv2
import numpy as np
import os
import sys
import time
import tensorflow as tf

from distutils.version import StrictVersion

try:
  if StrictVersion(tf.__version__) < StrictVersion('2.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v2.x.')
except:
  pass

# Path to label and frozen detection graph. This is the actual model that is used for the object detection.
parser = argparse.ArgumentParser(description='object detection tester, using image file')
parser.add_argument('-l', '--labels', default='./models/coco-labels-paper.txt', help="default: './models/coco-labels-paper.txt'")
parser.add_argument('-m', '--model', default='./models/centernet_hg104_512x512_coco17_tpu-8/saved_model/', help="default: './models/centernet_hg104_512x512_coco17_tpu-8/saved_model/'")
parser.add_argument('-i', '--input_image', default='', help="Input image file")
parser.add_argument('-o', '--output_image', default='', help="Output Image file")

args = parser.parse_args()

detection_graph = tf.Graph()

mode = 'bbox'

colors = [
  (0, 0, 255),
  (0, 64, 255),
  (0, 128, 255),
  (0, 192, 255),
  (0, 255, 255),
  (0, 255, 192),
  (0, 255, 128),
  (0, 255, 64),
  (0, 255, 0),
  (64, 255, 0),
  (128, 255, 0),
  (192, 255, 0),
  (255, 255, 0),
  (255, 192, 0),
  (255, 128, 0),
  (255, 64, 0),
  (255, 0, 0),
  (255, 0, 64),
  (255, 0, 128),
  (255, 0, 192),
  (255, 0, 255),
  (192, 0, 255),
  (128, 0, 255),
  (64, 0, 255),
]


def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)


def mosaic_area(src, x_min, y_min, x_max, y_max, ratio=0.1):
    dst = src.copy()
    dst[y_min:y_max, x_min:x_max] = mosaic(dst[y_min:y_max, x_min:x_max], ratio)
    return dst

# Load a saved model into memory.
print('Loading graph...')
DEFAULT_FUNCTION_KEY = "serving_default"
loaded = tf.saved_model.load(args.model)
inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]
print('Graph is loaded')


def run_inference_for_single_image(image, graph):
  # Run inference
  tensor = tf.convert_to_tensor(image)
  output_dict = inference_func(tensor)

  # all outputs are tensor, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict['detection_classes'][0].numpy()
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0].numpy()
  output_dict['detection_scores'] = output_dict['detection_scores'][0].numpy()

  return output_dict

count_max = 0

if __name__ == '__main__':
  count = 0

  labels = ['blank']
  with open(args.labels,'r') as f:
    for line in f:
      labels.append(line.rstrip())

  img = cv2.imread(args.input_image)
  #img_bgr = cv2.resize(img, (300,  300))
  width = 1920
  h, w = img.shape[:2]
  height = round(h * (width / w))
  img_bgr = cv2.resize(img, (width,  height))

  # convert bgr to rgb
  image_np = img_bgr[:,:,::-1]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  start = time.time()
  output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
  elapsed_time = time.time() - start

  for i in range(output_dict['num_detections']):
    class_id = output_dict['detection_classes'][i].astype(np.int)
    if class_id < len(labels):
      label = labels[class_id]
    else:
      label = 'unknown'

    detection_score = output_dict['detection_scores'][i]

    if detection_score > 0.5:
        # Define bounding box
        h, w, c = img.shape
        box = output_dict['detection_boxes'][i] * np.array( \
          [h, w,  h, w])
        box = box.astype(np.int)

        if mode == 'bbox':
          class_id = class_id % len(colors)
          color = colors[class_id]

          # Draw bounding box
          cv2.rectangle(img, \
          #  (box[1], box[0]), (box[3], box[2]), color, 3)
            (box[1], box[0]), (box[3], box[2]), (255, 255, 255), -1)

          font_size = 1.0
          box_w = box[3] - box[1]
          box_y = box[2] - box[0]
          if box_w < box_y :
            font_size = font_size * ( box_w / 40.0 )
          else :
            font_size = font_size * ( box_y / 40.0 )

          # Put label near bounding box
          #information = '%s: %.1f%%' % (label, output_dict['detection_scores'][i] * 100.0)
          information = '%s' % (label)
          #cv2.putText(img, information, (box[1] + 17, box[2] - 17), \
          #  cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 1, cv2.LINE_AA)
          cv2.putText(img, information, (box[1] + 15, box[2] - 15), \
            cv2.FONT_HERSHEY_TRIPLEX, font_size, (0, 0, 0), 1, cv2.LINE_AA)
          #  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
        elif mode == 'mosaic':
          img = mosaic_area(img, box[1], box[0], box[3], box[2], ratio=0.05)

  cv2.imwrite(args.output_image, img)
