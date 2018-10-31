from __future__ import division
import os
import numpy as np
import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle, csv
import sys
import time
import datetime

sys.path.append('..')
from utils import *
from model import UNet3D

# # FLAGS
flags = tf.app.flags
flags.DEFINE_integer("epoch", 1, "Epoch to train [4]")
flags.DEFINE_string("train_patch_dir", "/Users/nikhil/code/git_repos/skull_seg/test_output", "Directory of the training data [patches]")
flags.DEFINE_bool("split_train", False, "Whether to split the train data into train and val [False]")
flags.DEFINE_string("train_data_dir", "/Users/nikhil/code/git_repos/skull_seg/test_input/", "Directory of the train data [../BraTS17TrainingData]")
flags.DEFINE_string("deploy_data_dir", "/Users/nikhil/code/git_repos/skull_seg/test_input/", "Directory of the test data [../BraTS17ValidationData]")
flags.DEFINE_string("deploy_output_dir", "/Users/nikhil/code/git_repos/skull_seg/valid_input/", "Directory name of the output data [output]")
flags.DEFINE_string("train_csv", "../BraTS17TrainingData/survival_data.csv", "CSV path of the training data")
flags.DEFINE_string("deploy_csv", "../BraTS17ValidationData/survival_evaluation.csv", "CSV path of the validation data")
flags.DEFINE_integer("batch_size", 20, "Batch size [1]")
flags.DEFINE_integer("seg_features_root", 8, "Number of features in the first filter in the seg net [48]")
flags.DEFINE_integer("survival_features", 8, "Number of features in the survival net [16]")
flags.DEFINE_integer("conv_size", 3, "Convolution kernel size in encoding and decoding paths [3]")
flags.DEFINE_integer("layers", 1, "Encoding and deconding layers [3]")
flags.DEFINE_string("loss_type", "cross_entropy", "Loss type in the model [cross_entropy]")
flags.DEFINE_float("dropout", 0.5, "Drop out ratio [0.5]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "logs", "Directory name to save logs [logs]")
flags.DEFINE_boolean("train", True, "True for training, False for deploying [False]")
flags.DEFINE_boolean("run_seg", True, "True if run segmentation [True]")
flags.DEFINE_boolean("run_survival", False, "True if run survival prediction [True]")
FLAGS = flags.FLAGS

# pp = pprint.PrettyPrinter()
# pp.pprint(flags.FLAGS.__flags)

# Train
all_train_paths = []
# for dirpath, dirnames, files in os.walk(FLAGS.train_data_dir):
#     if os.path.basename(dirpath)[0:7] == 'Brats17':
#         all_train_paths.append(dirpath)

subject_dirs = next(os.walk(FLAGS.train_data_dir))[1]
for d in subject_dirs: 
    all_train_paths.append(os.path.join(FLAGS.train_data_dir,d))

training_paths = [os.path.join(FLAGS.train_patch_dir, name) for name in os.listdir(FLAGS.train_patch_dir)
                  if '.DS' not in name]
testing_paths = None

training_ids = [os.path.basename(i) for i in training_paths]
training_survival_paths = []
testing_survival_paths = None
training_survival_data = {}
testing_survival_data = None

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

start_time = datetime.datetime.now()
print(start_time)

if FLAGS.run_seg:
    run_config = tf.ConfigProto()
    with tf.Session(config=run_config) as sess:
    #with tf.Session() as sess:
    
        unet = UNet3D(sess, checkpoint_dir=FLAGS.checkpoint_dir, log_dir=FLAGS.log_dir, training_paths=training_paths,
                      testing_paths=testing_paths, batch_size=FLAGS.batch_size, layers=FLAGS.layers,
                      features_root=FLAGS.seg_features_root, conv_size=FLAGS.conv_size,
                      dropout=FLAGS.dropout, loss_type=FLAGS.loss_type)

        if FLAGS.train:
            model_vars = tf.trainable_variables()
            slim.model_analyzer.analyze_vars(model_vars, print_info=True)

            train_config = {}
            train_config['epoch'] = FLAGS.epoch

            unet.train(train_config)
        
        print('Training complete')
        print('\nTesting trained model with actual MR image')
        if not os.path.exists(FLAGS.deploy_output_dir):
            os.makedirs(FLAGS.deploy_output_dir)

        unet.deploy(FLAGS.deploy_data_dir, FLAGS.deploy_output_dir)
        print('Test complete. Output at {}'.format(FLAGS.deploy_output_dir))
    tf.reset_default_graph()

end_time = datetime.datetime.now()
print('start: {}, end: {}'.format(start_time, end_time))


