import argparse
import glob
import io
import os
import random
import math
import cv2
import numpy
import numpy as np
import tensorflow as tf
import datetime

from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from tensorflow.contrib.tensorboard.plugins import projector


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths. Change as per your dir structures .
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '/Users/sofwath/dev/chopstranslate/labels/thaana.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, '/Users/sofwath/dev/chopstranslate/fonts')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '/Users/sofwath/dev/chopstranslate/image-data')
DEFAULT_LABEL_CSV = os.path.join(SCRIPT_PATH, '/Users/sofwath/dev/chopstranslate/image-data/labels-map.csv')
DEFAULT_TFR_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '/Users/sofwath/dev/chopstranslate/tfrecords-output')
DEFAULT_MODEL_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'saved-model')

DEFAULT_NUM_SHARDS_TRAIN = 3
DEFAULT_NUM_SHARDS_TEST = 1

# Number of random distortion images to generate per font and akuru.
DISTORTION_COUNT = 512

# Width and height of the initial image.
IMAGE_WIDTH = 128   
IMAGE_HEIGHT = 128

MODEL_NAME = 'thaana_tensorflow'
MODEL_IMAGE_WIDTH = 64
MODEL_IMAGE_HEIGHT = 64

DEFAULT_NUM_TRAIN_STEPS = 20000
BATCH_SIZE = 100

def get_image(files, num_classes):
    
    # Convert filenames to a queue for an input pipeline.
    file_queue = tf.train.string_input_producer(files)

    # Create object to read TFRecords.
    reader = tf.TFRecordReader()

    # Read the full set of features for a single akuru.
    key, example = reader.read(file_queue)

    # Parse the akuru to get a dict mapping feature keys to tensors.
    # image/class/label: integer denoting the index in a classification layer.
    # image/encoded: string containing JPEG encoded image
    features = tf.parse_single_example(
        example,
        features={
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value='')
        })

    label = features['image/class/label']
    image_encoded = features['image/encoded']

    # Decode the JPEG.
    image = tf.image.decode_jpeg(image_encoded, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, [MODEL_IMAGE_WIDTH*MODEL_IMAGE_HEIGHT])

    # Represent the label as a one hot vector.
    label = tf.stack(tf.one_hot(label, num_classes))
    return label, image

def export_model(model_output_dir, input_node_names, output_node_name):
    
    name_base = os.path.join(model_output_dir, MODEL_NAME)
    frozen_graph_file = os.path.join(model_output_dir,
                                     'frozen_' + MODEL_NAME + '.pb')
    freeze_graph.freeze_graph(
        name_base + '.pbtxt', None, False, name_base + '.chkp',
        output_node_name, "save/restore_all", "save/Const:0",
        frozen_graph_file, True, ""
    )

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(frozen_graph_file, "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    optimized_graph_file = os.path.join(model_output_dir,
                                        'optimized_' + MODEL_NAME + '.pb')
    with tf.gfile.FastGFile(optimized_graph_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Inference optimized graph saved at: " + optimized_graph_file)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weight')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')

def train(label_file, tfrecords_dir, model_output_dir, num_train_steps):
    
    labels = io.open(label_file, 'r', encoding='utf-8').read().splitlines()
    num_classes = len(labels)

    # Define names so we can later reference specific nodes for when we use
    # the model for inference later.
    input_node_name = 'input'
    keep_prob_node_name = 'keep_prob'
    output_node_name = 'output'

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    print('Processing data...')

    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'train')
    train_data_files = tf.gfile.Glob(tf_record_pattern)
    label, image = get_image(train_data_files, num_classes)

    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'test')
    test_data_files = tf.gfile.Glob(tf_record_pattern)
    tlabel, timage = get_image(test_data_files, num_classes)

    # Associate objects with a randomly selected batch of labels and images.
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=BATCH_SIZE,
        capacity=2000,
        min_after_dequeue=1000)

    # Do the same for the testing data.
    timage_batch, tlabel_batch = tf.train.batch(
        [timage, tlabel], batch_size=BATCH_SIZE,
        capacity=2000)

    # Create the model!

    # Placeholder to feed in image data.
    x = tf.placeholder(tf.float32, [None, MODEL_IMAGE_WIDTH*MODEL_IMAGE_HEIGHT],
                       name=input_node_name)
    # Placeholder to feed in label data. Labels are represented as one_hot
    # vectors.
    y_ = tf.placeholder(tf.float32, [None, num_classes])

    # Reshape the image back into two dimensions so we can perform convolution.
    x_image = tf.reshape(x, [-1, MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT, 1])

    # First convolutional layer. 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1],
                           padding='SAME')
    h_conv1 = tf.nn.relu(x_conv1 + b_conv1)

    # Max-pooling.
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    # Second convolutional layer. 64 feature maps.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    x_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1],
                           padding='SAME')
    h_conv2 = tf.nn.relu(x_conv2 + b_conv2)

    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    # Third convolutional layer. 128 feature maps.
    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    x_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1],
                           padding='SAME')
    h_conv3 = tf.nn.relu(x_conv3 + b_conv3)

    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer. Here we choose to have 1024 neurons in this layer.
    h_pool_flat = tf.reshape(h_pool3, [-1, 8*8*128])
    W_fc1 = weight_variable([8*8*128, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

    # Dropout layer. This helps fight overfitting.
    keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Classification layer.
    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])
    yy = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Layer 4 
    y = tf.nn.softmax(yy,name='softmax') #chops

    # This isn't used for training, but for when using the saved model.
    tf.nn.softmax(yy, name=output_node_name)

    # Define our loss.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=yy) # change to softmax_cross_entropy_with_logits if results not accurate 
    )

    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    # Define accuracy.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    saver = tf.train.Saver()

    tf.summary.scalar('accuracy', accuracy)
    time_string = datetime.datetime.now().isoformat()
    experiment_name = f"thaana_{time_string}"
    merged = tf.summary.merge_all()
    

    with tf.Session() as sess:
        # Initialize the variables.
        sess.run(tf.global_variables_initializer())

        # Initialize the queue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        checkpoint_file = os.path.join(model_output_dir, MODEL_NAME + '.chkp')

        # Save the graph definition to a file.
        tf.train.write_graph(sess.graph_def, model_output_dir,
                             MODEL_NAME + '.pbtxt', True)

        train_writer = tf.summary.FileWriter(f'./train/{experiment_name}', sess.graph)
        test_writer  = tf.summary.FileWriter(f'./test/{experiment_name}', sess.graph)

        for step in range(num_train_steps):
            # Get a random batch of images and labels.
            train_images, train_labels = sess.run([image_batch, label_batch])

            # Perform the training step, feeding in the batches.
            sess.run(train_step, feed_dict={x: train_images, y_: train_labels,
                                            keep_prob: 0.5})

            # Print the training accuracy every 100 iterations.
            if step % 100 == 0:
                train_accuracy = sess.run(
                    accuracy,
                    feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0} # ,keep_prob: 1.0
                )
                print("Step %d, Training Accuracy %g" %
                      (step, float(train_accuracy)))


            # Every 10,000 iterations, we save a checkpoint of the model.
            if step % 10000 == 0:
                saver.save(sess, checkpoint_file, global_step=step)

        # Save a checkpoint after training has completed.
        saver.save(sess, checkpoint_file)

        # Get number of samples in test set.
        sample_count = 0
        for f in test_data_files:
            sample_count += sum(1 for _ in tf.python_io.tf_record_iterator(f))

        # See how model did by running the testing set through the model.
        print('Testing model...')

        # We will run the test set through batches and sum the total number
        # of correct predictions.
        num_batches = int(sample_count/BATCH_SIZE) or 1
        total_correct_preds = 0

        # Define a different tensor operation for summing the correct
        # predictions.
        accuracy2 = tf.reduce_sum(correct_prediction)
        for step in range(num_batches):
            image_batch2, label_batch2 = sess.run([timage_batch, tlabel_batch])
            acc = sess.run(accuracy2, feed_dict={x: image_batch2,
                                                 y_: label_batch2,
                                                 keep_prob: 1.0})
            total_correct_preds += acc

        accuracy_percent = total_correct_preds/(num_batches*BATCH_SIZE)
        print("Testing Accuracy {}".format(accuracy_percent))

        export_model(model_output_dir, [input_node_name, keep_prob_node_name],
                     output_node_name)

        # Stop queue threads and close session.
        coord.request_stop()
        coord.join(threads)
        sess.close()


def cropImage (im,file_path,labels_csv,character):
    
    im = im.convert("L")
    im = numpy.array(im)

    im_gray = im #cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # cv2.GaussianBlur(im, (5, 5), 0)

    #cv2.imshow('gray_image',im_gray) 
    #cv2.waitKey(0)       

    ret, im_th = cv2.threshold(im_gray, 180, 255, cv2.THRESH_BINARY) #cv2.THRESH_BINARY
    image, ctrs, hier = cv2.findContours(im_th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    for rect in rects:
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1) 
        # Make the rectangular region around the akuru
        leng = int(rect[3] * 1.6) #1.6
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # Resize the image and save
        roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_CUBIC)  # cv.INTER_LINEAR   cv2.INTER_AREA   
        cv2.imwrite(file_path,roi)
        labels_csv.write(u'{},{}\n'.format(file_path, character))

        #cv2.imshow('new_image',roi) 
        #cv2.waitKey(0)       

def generate_thaana_images(label_file, fonts_dir, output_dir):
    
    with io.open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    image_dir = os.path.join(output_dir, 'thaana-images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Get a list of the fonts.
    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

    labels_csv = io.open(os.path.join(output_dir, 'labels-map.csv'), 'w',
                         encoding='utf-8')

    total_count = 0
    prev_count = 0
    for character in labels:
        # Print image count roughly every 5000 images.
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} images generated...'.format(total_count))

        for font in fonts:
            total_count += 1
            image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
            fontu = font
            filename_w_ext = os.path.basename(fontu)
            filename, file_extension = os.path.splitext(filename_w_ext)
            font = ImageFont.truetype(font, 48)
            drawing = ImageDraw.Draw(image)
            #w, h = drawing.textsize(character, font=font)
            text_size = drawing.textsize(character, font=font) # the size of the text box!
            x = (IMAGE_WIDTH / 2) - (text_size[0] / 2)
            y = (IMAGE_HEIGHT / 2) - (text_size[0] / 2)
            drawing.text((x, y), character[0][0], font=font, fill=255)

            # print (character.strip())
            file_string = 'thaana_{}{}.jpeg'.format(total_count,filename)
            file_path = os.path.join(image_dir, file_string)
            
            cropImage(image,file_path,labels_csv,character)

            for i in range(DISTORTION_COUNT):
                total_count += 1
                file_string = 'thaana_{}{}.jpeg'.format(total_count,filename)
                file_path = os.path.join(image_dir, file_string)
                arr = numpy.array(image)

                distorted_array = elastic_distort(
                    arr, alpha=random.randint(30, 32), # alpha=random.randint(30, 36),
                    sigma=random.randint(5,6) # 5,6
                )
                distorted_image = Image.fromarray(distorted_array)
                cropImage(distorted_image,file_path,labels_csv,character)

    print('Done generating {} images.'.format(total_count))
    labels_csv.close()


def elastic_distort(image, alpha, sigma):

    random_state = numpy.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFRecordsConverter(object):

    def __init__(self, labels_csv, label_file, output_dir,
                 num_shards_train, num_shards_test):

        self.output_dir = output_dir
        self.num_shards_train = num_shards_train
        self.num_shards_test = num_shards_test

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Get lists of images and labels.
        self.filenames, self.labels = \
            self.process_image_labels(labels_csv, label_file)

        # Counter for total number of images processed.
        self.counter = 0

    def process_image_labels(self, labels_csv, label_file):
        
        labels_csv = io.open(labels_csv, 'r', encoding='utf-8')
        labels_file = io.open(label_file, 'r',
                              encoding='utf-8').read().splitlines()

        # Map akuru to indices.
        label_dict = {}
        count = 0
        for label in labels_file:
            label_dict[label] = count
            count += 1

        # Build the lists.
        images = []
        labels = []
        for row in labels_csv:
            file, label = row.strip().split(',')
            images.append(file)
            labels.append(label_dict[label])

        # Randomize the order of all the images/labels.
        shuffled_indices = list(range(len(images)))
        random.seed(12121)
        random.shuffle(shuffled_indices)
        filenames = [images[i] for i in shuffled_indices]
        labels = [labels[i] for i in shuffled_indices]

        return filenames, labels

    def write_tfrecords_file(self, output_path, indices):

        writer = tf.python_io.TFRecordWriter(output_path)
        for i in indices:
            filename = self.filenames[i]
            label = self.labels[i]
            with tf.gfile.FastGFile(filename, 'rb') as f:
                im_data = f.read()

            example = tf.train.Example(features=tf.train.Features(feature={'image/class/label': _int64_feature(label),'image/encoded': _bytes_feature(tf.compat.as_bytes(im_data))}))
            writer.write(example.SerializeToString())
            self.counter += 1
            if not self.counter % 1000:
                print('Processed {} images...'.format(self.counter))
        writer.close()

    def convert(self):
        
        num_files_total = len(self.filenames)

        # Allocate about 5 percent of images to testing
        num_files_test = int(num_files_total * .5)

        num_files_train = num_files_total - num_files_test

        print('Processing training set TFRecords...')

        files_per_shard = int(math.ceil(num_files_train /
                                        self.num_shards_train))
        start = 0
        for i in range(1, self.num_shards_train):
            shard_path = os.path.join(self.output_dir,
                                      'train-{}.tfrecords'.format(str(i)))
            # Get a subset of indices to get only a subset of images/labels for
            # the current shard file.
            file_indices = np.arange(start, start+files_per_shard, dtype=int)
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, file_indices)

        # The remaining images will go in the final shard.
        file_indices = np.arange(start, num_files_train, dtype=int)
        final_shard_path = os.path.join(self.output_dir,'train-{}.tfrecords'.format(str(self.num_shards_train)))
        self.write_tfrecords_file(final_shard_path, file_indices)

        print('Processing testing TFRecords set...')

        files_per_shard = math.ceil(num_files_test / self.num_shards_test)
        start = num_files_train
        for i in range(1, self.num_shards_test):
            shard_path = os.path.join(self.output_dir,'test-{}.tfrecords'.format(str(i)))
            file_indices = np.arange(start, start+files_per_shard, dtype=int)
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, file_indices)

        # The remaining images will go in the final shard.
        file_indices = np.arange(start, num_files_total, dtype=int)
        final_shard_path = os.path.join(self.output_dir,'test-{}.tfrecords'.format(str(self.num_shards_test)))
        self.write_tfrecords_file(final_shard_path, file_indices)

        print('\nProcessed {} total images...'.format(self.counter))
        print('Number of training : {}'.format(num_files_train))
        print('Number of testing : {}'.format(num_files_test))
        print('TFRecords files saved to {}'.format(self.output_dir))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing labels.')
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of Thaana fonts.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and '
                             'label CSV file.')
    args = parser.parse_args()
    generate_thaana_images(args.label_file, args.fonts_dir, args.output_dir)
    converter = TFRecordsConverter(DEFAULT_LABEL_CSV, DEFAULT_LABEL_FILE, DEFAULT_TFR_OUTPUT_DIR, DEFAULT_NUM_SHARDS_TRAIN, DEFAULT_NUM_SHARDS_TEST)
    converter.convert()
    train(args.label_file, DEFAULT_TFR_OUTPUT_DIR,DEFAULT_MODEL_OUTPUT_DIR, DEFAULT_NUM_TRAIN_STEPS)
