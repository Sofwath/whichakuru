#!/usr/bin/env python

import argparse
import io
import os
import sys
import cv2
import numpy as np
import tensorflow as tf

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default paths.
DEFAULT_LABEL_FILE = os.path.join(
    SCRIPT_PATH, '/Users/sofwath/dev/chopstranslate/labels/thaana.txt'
)
DEFAULT_GRAPH_FILE = os.path.join(
    SCRIPT_PATH, '/Users/sofwath/dev/chopstranslate/saved-model/optimized_thaana_tensorflow.pb'
)

def cropImage (im):

    im = cv2.imread(im)
    #im = im.convert("L")
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (1, 1), 0)
    
    #im = np.array(im)
    #im_gray = im #cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # cv2.GaussianBlur(im, (5, 5), 0)

    #cv2.imshow('image',im_gray) 
    #cv2.waitKey(0)       

    ret, im_th = cv2.threshold(im_gray, 180, 255, cv2.THRESH_BINARY_INV) #cv2.THRESH_BINARY
    image, ctrs, hier = cv2.findContours(im_th,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    if (len(rects) < 6):
        for rect in rects:
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
            # Make the rectangular region around the akuru
            leng = int(rect[3] * 1.6) #1.6
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
            # Resize the image and save
            roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_CUBIC)  # cv.INTER_LINEAR   cv2.INTER_AREA   
            args.image = roi
            classify(roi,DEFAULT_LABEL_FILE,DEFAULT_GRAPH_FILE)
    else:
        for rect in rects:
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
            # Make the rectangular region around the akuru
            leng = int(rect[3] * 1.6) #1.6
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
            # Resize the image and save
            roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_CUBIC)  # cv.INTER_LINEAR   cv2.INTER_AREA  
        print ('too many objects in image')

    cv2.imshow('gray_image_roi',im) 
    cv2.waitKey(0)

def read_image(file):
    """Read an image file and convert it into a 1-D floating point array."""
    file_content = tf.read_file(file)
    image = tf.image.decode_jpeg(file_content, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, [64*64])
    return image


def classify(im,DEFAULT_LABEL_FILE,DEFAULT_GRAPH_FILE):
    
    labels = io.open(DEFAULT_LABEL_FILE,'r', encoding='utf-8').read().splitlines()

    # Load graph and parse file.
    with tf.gfile.GFile(DEFAULT_GRAPH_FILE, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='thaana-model',
            op_dict=None,
            producer_op_list=None
        )

    # Get relevant nodes.
    x = graph.get_tensor_by_name('thaana-model/input:0')
    y = graph.get_tensor_by_name('thaana-model/output:0')
    keep_prob = graph.get_tensor_by_name('thaana-model/keep_prob:0')

    #image = read_image(args.image)
    image = im

    sess = tf.InteractiveSession()
    image_array = image # sess.run(image)

    with tf.Session(graph=graph) as graph_sess:
        predictions = graph_sess.run(y, feed_dict={x: image_array,keep_prob: 1.0})
        prediction = predictions[0]

    # Get the indices that would sort the array, then only get the indices that
    # correspond to the top 5 predictions.
    sorted_indices = prediction.argsort()[::-1][:1]
    for index in sorted_indices:
        label = labels[index]
        confidence = prediction[index]
        print('%s' % (label))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str,
                        help='Image to pass to model for classification.')
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--graph-file', type=str, dest='graph_file',
                        default=DEFAULT_GRAPH_FILE,
                        help='The saved model graph file to use for '
                             'classification.')
    args = parser.parse_args()
    cropImage (args.image)
    #classify(parser.parse_args())