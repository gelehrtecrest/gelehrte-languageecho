# coding: utf-8
# Object Detection Demo
import argparse
from queue import Empty
import cv2
import numpy as np
import os
import sys
import time
import tensorflow as tf
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont


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
parser.add_argument('-b', '--base_score', default='1', help="Base Score")
parser.add_argument('-s', '--sentence', default='y', help="Create the sentences")
parser.add_argument('-sd', '--similar_delete', default='y', help="Delete similar characters")
parser.add_argument('-c', '--cut', default='n', help="Cut letters")

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

#フォントの指定
font_name = '..\\..\\fonts\\NotoSansJP-Bold.otf'
#font_name = '..\\..\\fonts\\NieR-Regular.ttf'


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

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def cv2_putText_2(img, text, org, fontFace, fontScale, color, max_width):
    x, y = org
    b, g, r = color
    colorRGB = (r, g, b)
    imgPIL = cv2pil(img)
    draw = ImageDraw.Draw(imgPIL)
    fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
    w, h = draw.textsize(text, font = fontPIL)
    if w > max_width:
      fontScale2 = int( fontScale * ( max_width / w ) )
      fontPIL = ImageFont.truetype(font = fontFace, size = fontScale2)
      w, h = draw.textsize(text, font = fontPIL)
    draw.text(xy = (x,y-h), text = text, fill = colorRGB, font = fontPIL)
    imgCV = pil2cv(imgPIL)
    return imgCV

count_max = 0

def workingfilewrite(filename, fix, message):
  filepath = 'D:\workspace\github\gelehrte-languageecho\\test\work\\' + filename + '-' + fix + '.txt'
  f = open(filepath, 'w')
  f.write(message)

  now = datetime.now()
  d = now.strftime('%Y/%m/%d %H:%M:%S')
  f.write('\n' + d)

  f.close()


def is_near(word, box):
  #X軸の小さい順にならんでいるので、一番近い可能性があるのは最後の文字
  word_last_letter = word[-1]
  word_last_letter_box = word_last_letter["box"]

  #Y軸チェック
  box_height = box[2] - box[0]
  #Y軸でboxの高さの小さいほうから1/4より下に、word_last_letterのY軸の下部が入っていないとFalse
  if word_last_letter_box[2] < box[0] + box_height / 4 :
    return False
  #Y軸でboxの高さの高いほうから1/4より上に、word_last_letterのY軸の上部が入っていないとFalse
  if word_last_letter_box[0] > box[2] - box_height / 4 :
    return False

  #X軸チェック
  distance = distance_word(word, box)
  #distanceがboxの幅の1/6以上である場合はFalse
  if distance > (box[3] - box[1]) / 6 and distance > (word_last_letter_box[3] - word_last_letter_box[1]) / 6:
    return False

  return True

def distance_word(word, box):
  #X軸の差だけ見ている
  #X軸の小さい順に見ているので、単純にwordの最後の文字との差だけでいい
  word_last_letter = word[-1]
  word_last_letter_box = word_last_letter["box"]
  
  distance = box[1] - word_last_letter_box[3]
  if distance < 0:
    return 0

  return distance

def new_word_list_number(word_list):
  return len(word_list) + 1

def add_word_2_word_list(word_list, label, box):
  letter = {"label": label, "box": box, "word_number": None}
  near_word = None
  near_distance = -1
  for word_number in word_list.keys():
    word = word_list[word_number]
    #より距離が近いletterを持つwordを探す
    if is_near(word, box):
      new_distance = distance_word(word, box)
      if near_distance < 0 or near_distance > new_distance:
        near_word = word
        near_distance = new_distance

  if near_word is None:
    #距離が近いletterが見つからなかった場合
    #単体letterで一つのwordとして認識する
    word_number = new_word_list_number(word_list)
    letter["word_number"] = word_number
    new_word = [letter]
    word_list[word_number] = new_word
  else :
    #距離が近いletterを持つwordが見つかった場合
    word_number = near_word[0].get("word_number")
    word = word_list[word_number]
    #追加するletterのword_numberを更新
    letter["word_number"] = word_number
    #そのwordに追加したいletterを追加する
    word.append(letter)
    #wordをwordlistに再登録
    word_list[word_number] = word

  return word_list


def add_letter_2_letter_list(letter_list, label, box, detection_score):
  new_letter_list = []
  new_letter_x = box[1]
  new_letter = {"x": new_letter_x, "label": label, "box": box, "detection_score": detection_score}
  append_flag = True
  print("----------------------------------------")
  print(new_letter)
  print_letter_list(letter_list)
  #まず予め、似ている文字のが存在する場合はscoreの高い文字を残す
  similar_flag = False
  tmp_letter_list = []
  for letter in letter_list:
    if is_similar_letter(letter, new_letter):
      print("is_similar_letter-------------")
      print(letter)
      print(new_letter)
      similar_flag = True
      if letter["detection_score"] > new_letter["detection_score"]:
        tmp_letter_list.append(letter)
      else:
        tmp_letter_list.append(new_letter)
    else:
      tmp_letter_list.append(letter)

  #print_letter_list(tmp_letter_list)

  if similar_flag:
    #似ている文字が存在する場合はtmp_letter_listをそのままnew_letter_listとする
    new_letter_list = tmp_letter_list
  else :
    for letter in tmp_letter_list:
      letter_x = letter["x"]
      if new_letter_x < letter_x and (append_flag):
        new_letter_list.append(new_letter)
        append_flag = False
      new_letter_list.append(letter)
    if (append_flag):
      new_letter_list.append(new_letter)
  print_letter_list(new_letter_list)
  return new_letter_list


def get_sentence_box(sentence):
  sentence_box = [-1, -1, -1, -1]
  for word in sentence:
    for letter in word:
      letter_box = letter["box"]
      if sentence_box[0] < 0 or sentence_box[0] > letter_box[0]:
        sentence_box[0] = letter_box[0]
      if sentence_box[1] < 0 or sentence_box[1] > letter_box[1]:
        sentence_box[1] = letter_box[1]
      if sentence_box[2] < 0 or sentence_box[2] < letter_box[2]:
        sentence_box[2] = letter_box[2]
      if sentence_box[3] < 0 or sentence_box[3] < letter_box[3]:
        sentence_box[3] = letter_box[3]

  return sentence_box

def get_word_box(word):
  word_box = [-1, -1, -1, -1]
  for letter in word:
    letter_box = letter["box"]
    if word_box[0] < 0 or word_box[0] > letter_box[0]:
      word_box[0] = letter_box[0]
    if word_box[1] < 0 or word_box[1] > letter_box[1]:
      word_box[1] = letter_box[1]
    if word_box[2] < 0 or word_box[2] < letter_box[2]:
      word_box[2] = letter_box[2]
    if word_box[3] < 0 or word_box[3] < letter_box[3]:
      word_box[3] = letter_box[3]

  return word_box

def is_near_sentence_lastword_2_word(sentence, word):
  #sentenceの最後のletterとwordの最初のletterのX軸の距離が、幅の平均より広ければFalse
  sentence_lastletter = sentence[-1][-1]
  word_firstletter = word[0]

  #sentenceの最後のletterの幅
  sentence_lastletter_width = sentence_lastletter["box"][3] - sentence_lastletter["box"][1]
  #wordの最初のletterの幅
  word_firstletter_width = word_firstletter["box"][3] - word_firstletter["box"][1]

  #距離
  distance = word_firstletter["box"][1] - sentence_lastletter["box"][3]

  #幅の平均
  width_2 = (sentence_lastletter_width + word_firstletter_width) / 2

  if distance > width_2:
    return False
  return True

def in_sentence(sentence_list, word):
  num = 0
  word_box = get_word_box(word)
  word_y_half = (word_box[2] - word_box[0]) / 2 + word_box[0]


  for sentence in sentence_list:
    sentence_box = get_sentence_box(sentence)
    if word_y_half > sentence_box[0] and word_y_half < sentence_box[2]:
      if is_near_sentence_lastword_2_word(sentence, word):
        return (sentence, num)
    num = num + 1

  return (None, -1)

    

def get_sentence_list(word_list):
  sentence_list = []
  for word_number in word_list.keys():
    word = word_list[word_number]
    (sentence, num) = in_sentence(sentence_list, word)
    if sentence is None:
      sentence_list.append([word])
    else :
      sentence.append(word)
      sentence_list[num] = sentence

  return sentence_list


#自分のsentenceの高さ中央より、sentenceの頂点が低い場合、sentenceの順番を前にする
def sort_sentence_list(sentence_list):
  #まず頂点と中央の高さを保管する
  sentence_height_data_list = []
  for sentence in sentence_list:
    sentence_box = get_sentence_box(sentence)
    height = sentence_box[0]
    center = (sentence_box[0] + sentence_box[2]) / 2
    sentence_height_data = {"sentence" : sentence, "height": height, "center": center}
    sentence_height_data_list.append(sentence_height_data)

  #ソートする
  return_sentence_height_data_list = []
  for sentence_height_data in sentence_height_data_list:
    flag = True
    if not return_sentence_height_data_list:
      return_sentence_height_data_list.append(sentence_height_data)
    else:
      tmp_sentence_height_data_list = []
      for tmp_sentence_height_data in return_sentence_height_data_list:
        if tmp_sentence_height_data["height"] > sentence_height_data["center"]:
          tmp_sentence_height_data_list.append(sentence_height_data)
          flag = False
        tmp_sentence_height_data_list.append(tmp_sentence_height_data)
      if flag:
        tmp_sentence_height_data_list.append(sentence_height_data)
      return_sentence_height_data_list = tmp_sentence_height_data_list

  #結果を保存する
  return_sentence_list = []
  for return_sentence_height_data in return_sentence_height_data_list:
    return_sentence_list.append(return_sentence_height_data["sentence"])

  return return_sentence_list

#一般的な判定基準
base_score_1 = 0.6
base_score_2 = 0.4
base_score_3 = 0.2
base_score_hard_1 = 0.6
#ILilは特に認識難しいので、特別扱い
veryLowScoreLabel_score = 0.5
veryLowScoreLabel = [
  "I",
  "i"
]
def is_veryLowScoreLabel(label):
  return (label in veryLowScoreLabel)
#認識が難しいラベルかどうか
lowScoreLabel_score = 0.5
lowScoreLabel = [
  "A",
  "B",
  "C",
  "D",
  "E",
  "L",
  "P",
  "Q",
  "V",
  "Y",
  "Z",
  "a",
  "b",
  "h",
  "l",
  "p",
  "q",
  "v",
  "y",
  "z"
  ]
def is_lowScoreLabel(label):
  return (label in lowScoreLabel)

#誤読が多いラベルかどうか
misreadingLabel_score = 0.6
misreadingLabel = [
  [
    "A",
    "a",
    "n",
    "y"
  ],
  [
    "C",
    "D",
    "G",
    "c",
    "g"
  ],
  [
    "d",
    "b"
  ],
  [
    "E",
    "e"
  ],
  [
    "F",
    "f"
  ],
  [
    "I",
    "L",
    "Q",
    "T",
    "W",
    "h",
    "i",
    "l",
    "q",
    "t",
    "w"
  ],
  [
    "O",
    "o",
    "0"
  ],
  [
    "R",
    "r"
  ],
  [
    "S",
    "s"
  ],
  [
    "T",
    "W",
    "t",
    "w"
  ],
  [
    "V",
    "Y",
    "v",
    "y"
  ]
  ]
def is_misreadingLabel(label):
  for misreadingLabel_list in misreadingLabel:
    if label in misreadingLabel_list:
      return True
  return False

def is_misreadingLabel_in_same_list(label1, label2):
  for misreadingLabel_list in misreadingLabel:
    if label1 in misreadingLabel_list and label2 in misreadingLabel_list:
      return True
  return False

#共通面積のしきい値
misreading_area = 0.7
same_area = 0.9
veryLowScore_area = 0.9
def is_similar_letter(letter1, letter2):
  if args.similar_delete == 'n':
    return False

  print("is_similar_check------")
  print(letter1)
  print(letter2)
  similar_area = same_area
  #誤読リストに入っているかどうかのチェック
  if is_misreadingLabel_in_same_list(letter1["label"], letter2["label"]):
    similar_area = misreading_area
  #特に難しいリストに入っているかどうかのチェック
  if is_veryLowScoreLabel(letter1["label"]) or is_veryLowScoreLabel(letter2["label"]):
    similar_area = veryLowScore_area
  #共通面積が、どちらかの面積のしきい値以上を占めている場合、似ているとする
  x_1 = 0
  x_2 = 0
  y_1 = 0
  y_2 = 0
  box1 = letter1["box"]
  box2 = letter2["box"]

  #全く重なりがない場合は先にFalseと返す
  if box1[3] < box2[1]:
    return False
  if box1[1] > box2[3]:
    return False
  if box2[2] < box2[0]:
    return False
  if box2[0] > box2[2]:
    return False

  #共通部分の座標取得
  if box1[1] < box2[1]:
    x_1 = box2[1]
  else:
    x_1 = box1[1]
  if box1[3] < box2[3]:
    x_2 = box1[3]
  else:
    x_2 = box2[3]
  if box1[0] < box2[0]:
    y_1 = box2[0]
  else:
    y_1 = box1[0]
  if box1[2] < box2[2]:
    y_2 = box1[2]
  else:
    y_2 = box2[2]
  #共通部分の面積
  common_area = (x_2 - x_1) * (y_2 - y_1)
  
  #各letterの面積を求める
  area1 = (box1[3] - box1[1]) * (box1[2] - box1[0])
  area2 = (box2[3] - box2[1]) * (box2[2] - box2[0])
  #面積の狭い方と比較する
  area = 0
  if area1 < area2:
    area = area1
  else:
    area = area2

  #print(common_area / area)
  #しきい値より高い場合、同じ場所とする
  print(common_area)
  print(area1)
  print(area2)
  print(common_area / area)
  if (common_area / area) > similar_area:
    return True
  return False

def print_letter_list(letter_list):
  for letter in letter_list:
    print(letter["label"], end='')
    print(",", end='')

  print("")

def start_languageecho():
  workingfilewrite('test', '01', 'モデル読み込みました')
  count = 0

  labels = ['blank']
  with open(args.labels,'r') as f:
    for line in f:
      labels.append(line.rstrip())

  img = cv2.imread(args.input_image)
  #img_bgr = cv2.resize(img, (300,  300))
  size = 1920
  h, w = img.shape[:2]

  font_scale = np.log10(1080 / h) + 1.0
  #font_size = 120.0 * font_scale


  #1920*1920のサイズ感で同じような面積に拡大
  if h < w:
    size_rate = int((w / h)**0.5)
    width = size * size_rate
    height = round(h * (width / w))
  else :
    size_rate = int((h / w)**0.5)
    height = size * size_rate
    width = round(w * (height / h))

  print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
  print(height)
  print(width)
  img_bgr = cv2.resize(img, (width,  height))

  # convert bgr to rgb
  image_np = img_bgr[:,:,::-1]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  start = time.time()
  output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
  elapsed_time = time.time() - start

  #全体の文字を一時的に保管する配列
  letter_list = []

  #base_score
  #判定しやすさを変える
  base_score = base_score_1
  if args.base_score == '2':
    base_score = base_score_2
  elif args.base_score == '3':
    base_score = base_score_3

  detection_score_base = base_score
  for i in range(output_dict['num_detections']):
    detection_score_label = detection_score_base
    class_id = output_dict['detection_classes'][i].astype(np.int)
    print("^^^^^^^^^^^^^^^^^^^^^^^")
    if class_id < len(labels):
      label = labels[class_id]
      print(label)
      if is_veryLowScoreLabel(label):
        detection_score_label = veryLowScoreLabel_score
      elif is_lowScoreLabel(label):
        detection_score_label = lowScoreLabel_score
      elif is_misreadingLabel(label):
        detection_score_label = misreadingLabel_score
    else:
      label = 'unknown'

    #判定しやすいbase_scoreにした場合、それに対応させる
    if detection_score_label > base_score:
      detection_score_label = base_score

    #ラベルによる重要度の区別をしない場合
    if args.base_score == 'h1':
      detection_score_label = base_score_hard_1

    detection_score = output_dict['detection_scores'][i]
    print(detection_score)
    if detection_score > detection_score_label:
        h, w, c = img.shape
        box = output_dict['detection_boxes'][i] * np.array( \
          [h, w,  h, w])
        box = box.astype(np.int)
        letter_list = add_letter_2_letter_list(letter_list, label, box, detection_score)


  if args.sentence == 'y':
    #全体の文字を入れる連想配列
    word_list = {}
    for letter in letter_list:
      print(letter["label"], end='')
      print(",", end='')
      word_list = add_word_2_word_list(word_list, letter["label"], letter["box"])
    print(" / ")


    sentence_list = get_sentence_list(word_list)
    sentence_list = sort_sentence_list(sentence_list)
    #先に塗りつぶす
    for sentence in sentence_list:
      sentence_box = get_sentence_box(sentence)
      cv2.rectangle(img, (sentence_box[1], sentence_box[0]), (sentence_box[3], sentence_box[2]), (255, 255, 255), -1)

    #次に文字を描く
    for sentence in sentence_list:
      text = ""
      for word in sentence:
        for letter in word:
          text += letter["label"]
        text += " "
      print(text)
      sentence_box = get_sentence_box(sentence)
      font_size = 0.3 * (sentence_box[2] - sentence_box[0]) * font_scale
      colorBGR = (0,0,0)
      img = cv2_putText_2(img = img,
                          text = text,
                          org = (sentence_box[1] + 5, sentence_box[2] - 5),
                          fontFace = font_name,
                          fontScale = int(font_size),
                          color = colorBGR,
                          max_width = sentence_box[3] - sentence_box[1]
                          )
  else :
    #文字だけで処理する
    #先に塗りつぶす
    for letter in letter_list:
      cv2.rectangle(img, (letter["box"][1], letter["box"][0]), (letter["box"][3], letter["box"][2]), (255, 255, 255), -1)
    for letter in letter_list:
      text = letter["label"]
      font_size = 0.3 * (letter["box"][2] - letter["box"][0]) * font_scale
      colorBGR = (0,0,0)
      img = cv2_putText_2(img = img,
                          text = text,
                          org = (letter["box"][1] + 5, letter["box"][2] - 5),
                          fontFace = font_name,
                          fontScale = int(font_size),
                          color = colorBGR,
                          max_width = letter["box"][3] - letter["box"][1]
                          )
  cv2.imwrite(args.output_image, img)

def start_languageecho_cut():
  print("start_languageecho_cut")
  workingfilewrite('test', '01', 'モデル読み込みました')
  count = 0

  labels = ['blank']
  with open(args.labels,'r') as f:
    for line in f:
      labels.append(line.rstrip())

  img = cv2.imread(args.input_image)
  #img_bgr = cv2.resize(img, (300,  300))
  size = 1920
  h, w = img.shape[:2]
  if h < w:
    if h < w * 2:
      if h < w * 4:
        width = size * 4
      else :
        width = size * 2
    else :
      width = size
    height = round(h * (width / w))
  else :
    if h * 2 > w:
      if h * 4 > w:
        height = size * 4
      else :
        height = size * 2
    else :
      height = size
    width = round(w * (height / h))
  img_bgr = cv2.resize(img, (width,  height))

  # convert bgr to rgb
  image_np = img_bgr[:,:,::-1]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  start = time.time()
  output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
  elapsed_time = time.time() - start

  num = 0
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
          num = num + 1
          base_filename = args.output_image
          output_filename = base_filename + str(num) + ".png"
          class_id = class_id % len(colors)
          color = colors[class_id]

          print(output_filename)
          print(box[0])
          print(box[1])
          print(box[2])
          print(box[3])

          #これで切り抜きはOK
          #ただ反転などは未実装これから実装する
          img1 = img[box[0] : box[2], box[1] : box[3]]
          cv2.imwrite(output_filename, img1)

          # Draw bounding box
          cv2.rectangle(img, \
            (box[1], box[0]), (box[3], box[2]), color, 3)

          font_size = 5.0
          box_w = box[3] - box[1]
          box_y = box[2] - box[0]
          if box_w < box_y :
            font_size = font_size * ( box_w / 40.0 )
          else :
            font_size = font_size * ( box_y / 40.0 )

          # Put label near bounding box
          information = '%s: %.1f%%' % (label, output_dict['detection_scores'][i] * 100.0)
          cv2.putText(img, information, (box[1] + 15, box[2] - 15), \
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
        elif mode == 'mosaic':
          img = mosaic_area(img, box[1], box[0], box[3], box[2], ratio=0.05)

  cv2.imwrite(args.output_image, img)



if __name__ == '__main__':
  if args.cut == 'y':
    start_languageecho_cut()
  else :
    start_languageecho()
