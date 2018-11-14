from __future__ import division
import tensorflow as tf
import os, re, time
import numpy as np
import pickle

from utils import *

def conv3d(input_, output_dim, f_size, is_training, scope='conv3d'):
    # TODO
    # add batch norm
    # net = slim.fully_connected(X, net_arch['l{}'.format(l+1)],normalizer_fn=slim.batch_norm,scope='fc{}'.format(l))
    
    with tf.variable_scope(scope) as scope:
        # VGG network uses two 3*3 conv layers to effectively increase receptive field
        w1 = tf.get_variable('w1', [f_size, f_size, f_size, input_.get_shape()[-1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1 = tf.nn.conv3d(input_, w1, strides=[1, 1, 1, 1, 1], padding='SAME')
        b1 = tf.get_variable('b1', [output_dim], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.bias_add(conv1, b1)
        #bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training, scope='bn1', decay=0.9,
        #                                  variables_collections=['bn_collections'])
        
        # default TF layer no contrib
        bn1 = tf.layers.batch_normalization(conv1, training=is_training)
        r1 = tf.nn.relu(bn1)
        
        ## Comment starts here -------
        w2 = tf.get_variable('w2', [f_size, f_size, f_size, output_dim, output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2 = tf.nn.conv3d(r1, w2, strides=[1, 1, 1, 1, 1], padding='SAME')
        b2 = tf.get_variable('b2', [output_dim], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.bias_add(conv2, b2)
                                        
        # bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training, scope='bn2', decay=0.9,
        #                                    zero_debias_moving_mean=True, variables_collections=['bn_collections'])
        
        # default TF layer no contrib
        bn2 = tf.layers.batch_normalization(conv2, training=is_training)
        r2 = tf.nn.relu(bn2)
        ## Comment ends here -------
        
        return r2
    
def deconv3d(input_, output_shape, f_size, is_training, scope='deconv3d'):
    with tf.variable_scope(scope) as scope:
        output_dim = output_shape[-1]
        w = tf.get_variable('w', [f_size, f_size, f_size, output_dim, input_.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        deconv = tf.nn.conv3d_transpose(input_, w, output_shape, strides=[1, f_size, f_size, f_size, 1], padding='SAME')
        
        # bn = tf.contrib.layers.batch_norm(deconv, is_training=is_training, scope='bn', decay=0.9,
        #                                   zero_debias_moving_mean=True, variables_collections=['bn_collections'])
        bn3 = tf.layers.batch_normalization(deconv, training=is_training)
        r = tf.nn.relu(bn3)
        
        return r
    
def crop_and_concat(x1, x2):
    x1_shape = x1.get_shape().as_list()
    x2_shape = x2.get_shape().as_list()
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 4) #tf.concat(4, [x1_crop, x2]) #tf.concat([x1_crop, x2], 4) # order depends on TF version

# def conv_relu(input_, output_dim, f_size, s_size, scope='conv_relu'):
#     with tf.variable_scope(scope) as scope:
#         w = tf.get_variable('w', [f_size, f_size, f_size, input_.get_shape()[-1], output_dim],
#                             initializer=tf.truncated_normal_initializer(stddev=0.1))
#         conv = tf.nn.conv3d(input_, w, strides=[1, s_size, s_size, s_size, 1], padding='VALID')
#         b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
#         conv = tf.nn.bias_add(conv, b)
#         r = tf.nn.relu(conv)
#         return r
    
class UNet3D(object):
    def __init__(self, sess, checkpoint_dir, log_dir, training_paths, testing_paths,
                 batch_size=1, layers=3, features_root=32, conv_size=3, dropout=0.5,
                 loss_type='cross_entropy', class_weights=None):
        self.sess = sess
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        self.training_paths = training_paths
        self.testing_paths = testing_paths
        self.nclass = 2

        image, _ = read_patch(os.path.join(self.training_paths[0], '0'),self.nclass)
        
        self.batch_size = batch_size
        self.patch_size = image.shape[:-1]
        self.patch_stride = 4 # Used in deploy
        self.channel = image.shape[-1]
        self.layers = layers
        self.features_root = features_root
        self.conv_size = conv_size
        self.dropout = dropout
        self.loss_type = loss_type
        self.class_weights = class_weights
        self.patches_per_image = len(os.listdir(self.training_paths[0]))
        
        self.build_model()
        
        self.saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection_ref('bn_collections'))
        
    def build_model(self):
        self.images = tf.placeholder(tf.float32, shape=[None, self.patch_size[0], self.patch_size[1], self.patch_size[2],
                                                        self.channel], name='images')
        self.labels = tf.placeholder(tf.float32, shape=[None, self.patch_size[0], self.patch_size[1], self.patch_size[2],
                                                        self.nclass], name='labels')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_ratio')
        
        conv_size = self.conv_size
        layers = self.layers

        deconv_size = 2
        pool_stride_size = 2
        pool_kernel_size = 3 # Use a larger kernel
        
        # Encoding path
        print('encoding...')
        connection_outputs = []
        for layer in range(layers):
            features = 2**layer * self.features_root
            if layer == 0:
                prev = self.images
            else:
                prev = pool
                
            conv = conv3d(prev, features, conv_size, is_training=self.is_training, scope='encoding' + str(layer))
            connection_outputs.append(conv)
            pool = tf.nn.max_pool3d(conv, ksize=[1, pool_kernel_size, pool_kernel_size, pool_kernel_size, 1],
                                    strides=[1, pool_stride_size, pool_stride_size, pool_stride_size, 1],
                                    padding='SAME')
        
        bottom = conv3d(pool, 2**layers * self.features_root, conv_size, is_training=self.is_training, scope='bottom')
        bottom = tf.nn.dropout(bottom, self.keep_prob)

        
        # Decoding path
        print('decoding...')
        for layer in range(layers):
            conterpart_layer = layers - 1 - layer
            features = 2**conterpart_layer * self.features_root
            if layer == 0:
                prev = bottom
            else:
                prev = conv_decoding
            
            shape = prev.get_shape().as_list()
            
            deconv_output_shape = [tf.shape(prev)[0], shape[1] * deconv_size, shape[2] * deconv_size,
                                   shape[3] * deconv_size, features]
            
            deconv = deconv3d(prev, deconv_output_shape, deconv_size, is_training=self.is_training,
                              scope='decoding_dconv' + str(conterpart_layer))
           
            # cc = crop_and_concat(connection_outputs[conterpart_layer], deconv)
            # conv_decoding = conv3d(cc, features, conv_size, is_training=self.is_training,
            #                        scope='decoding' + str(conterpart_layer))
            conv_decoding = conv3d(deconv, features, conv_size, is_training=self.is_training,
                                   scope='decoding_conv' + str(conterpart_layer))

        # print('conv_decoding shape: {}'.format(conv_decoding.get_shape()))    
        with tf.variable_scope('logits') as scope:
            w = tf.get_variable('w', [1, 1, 1, conv_decoding.get_shape()[-1], self.nclass],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            
            logits = tf.nn.conv3d(conv_decoding, w, strides=[1, 1, 1, 1, 1], padding='SAME')
            b = tf.get_variable('b', [self.nclass], initializer=tf.constant_initializer(0.0))
            logits = tf.nn.bias_add(logits, b)
        
        # print('logits shape: {}'.format(logits.get_shape()))    
        # print('labels shape: {}'.format(self.labels.get_shape()))    
        self.probs = tf.nn.softmax(logits)
        self.predictions = tf.argmax(self.probs, 4) # not sure why this is 4
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(self.labels, 4)), tf.float32)) # not sure why this is 4
                                  
        flat_logits = tf.reshape(logits, [-1, self.nclass])
        flat_labels = tf.reshape(self.labels, [-1, self.nclass])

        if self.class_weights is not None:
            class_weights = tf.constant(np.asarray(self.class_weights, dtype=np.float32))
            weight_map = tf.reduce_max(tf.multiply(flat_labels, class_weights), axis=1)
            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
            weighted_loss = tf.multiply(loss_map, weight_map)
            cross_entropy_loss = tf.reduce_mean(weighted_loss)
        else:
            cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,labels=flat_labels))

            #cross_entropy_loss = tf.reduce_mean(flat_logits)

        eps = 1e-5
        dice_loss = 0
        dice_value = 0
        for i in range(1, self.nclass):
            slice_prob = tf.squeeze(tf.slice(self.probs, [0, 0, 0, 0, i], [-1, -1, -1, -1, 1]), axis=4)
            slice_prediction = tf.cast(tf.equal(self.predictions, i), tf.float32)
            slice_label = tf.squeeze(tf.slice(self.labels, [0, 0, 0, 0, i], [-1, -1, -1, -1, 1]), axis=4)
            intersection_prob = tf.reduce_sum(tf.multiply(slice_prob, slice_label), axis=[1, 2, 3])
            intersection_prediction = tf.reduce_sum(tf.multiply(slice_prediction, slice_label), axis=[1, 2, 3])
            union = eps + tf.reduce_sum(slice_prediction, axis=[1, 2, 3]) + tf.reduce_sum(slice_label, axis=[1, 2, 3])
            dice_loss += tf.reduce_mean(tf.div(intersection_prob, union))
            dice_value += tf.reduce_mean(tf.div(intersection_prediction, union))
        dice_value = dice_value * 2.0 / (self.nclass - 1)
        dice_loss = 1 - dice_loss * 2.0 / (self.nclass - 1)
        self.dice = dice_value
        
        if self.loss_type == 'cross_entropy':
            self.loss = cross_entropy_loss
        elif self.loss_type == 'dice':
            self.loss = cross_entropy_loss + dice_loss
        else:
            raise ValueError("Unknown cost function: " + self.loss_type)
        
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        self.dice_summary = tf.summary.scalar('dice', self.dice)
            
    def train(self, config):
        #optimizer = AdamOptimizer().minimize(self.loss)
        print('train def')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        
        self.sess.run(tf.global_variables_initializer())
        
        train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'train'), self.sess.graph)
        if self.testing_paths is not None:
            test_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'test'))
            testing_orders = [(n, l) for n in range(len(self.testing_paths)) for l in range(self.patches_per_image)]
        
        merged = tf.summary.merge([self.loss_summary, self.accuracy_summary, self.dice_summary])
                
        counter = 0
        test_interval = 10
        ckpt_interval = 1000
        
        train_loss_list = []
        train_acc_list = []
        train_dice_list = []
        test_loss_list = []
        test_acc_list = []
        test_dice_list = []
       
        training_orders = [(n, l) for n in range(len(self.training_paths)) for l in range(self.patches_per_image)]
        print('channels and classes: {} {}'.format(self.channel,self.nclass))
        for epoch in range(config['epoch']):
            # Shuffle the orders
            epoch_training_orders = np.random.permutation(training_orders)
            # Go through all selected patches
            for f in range(len(epoch_training_orders) // self.batch_size):
                print('train order: {}'.format(f))
                patches = np.empty((self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], self.channel),
                                   dtype=np.float32)
                labels = np.empty((self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], self.nclass),
                                  dtype=np.float32)
                
                for b in range(self.batch_size):
                    order = epoch_training_orders[f * self.batch_size + b]
                    patches[b], labels[b] = read_patch(os.path.join(self.training_paths[order[0]], str(order[1])),self.nclass)

                # print('patch, label means: {}, {}'.format(np.mean(patches),np.mean(labels)))
                # print('reordered patches and labels: {} {}'.format(patches.shape, labels.shape))
                
                _, train_loss, train_acc, train_dice, summary = self.sess.run([optimizer, self.loss, self.accuracy, 
                                                                               self.dice, merged],
                                                       feed_dict = { self.images: patches,
                                                                     self.labels: labels,
                                                                     self.is_training: True,
                                                                     self.keep_prob: self.dropout })
                
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                train_dice_list.append(train_dice)
                
                # print('train forward pass complete')
                train_writer.add_summary(summary, counter)
                counter += 1
                if np.mod(counter, ckpt_interval) == 0:
                    self.save(counter)
                    
                # # Run test
                if self.testing_paths is not None and np.mod(counter, test_interval) == 0:
                    for b in range(self.batch_size):
                        order = testing_orders[np.random.choice(len(testing_orders))]
                        patches[b], labels[b] = read_patch(os.path.join(self.testing_paths[order[0]], str(order[1])),self.nclass)
                    
                    test_loss, test_acc, test_dice, summary = self.sess.run([self.loss, self.accuracy, self.dice, merged],
                                                       feed_dict = { self.images: patches,
                                                                     self.labels: labels,
                                                                     self.is_training: True,
                                                                     self.keep_prob: 1 })
                    
                    print(str(counter) + ":" + "train_loss: " + str(train_loss) + " test_loss: " + str(test_loss))
                    print(str(counter) + ":" + "train_acc: " + str(train_acc) + " test_acc: " + str(test_acc))
                    print(str(counter) + ":" + "train_dice: " + str(train_dice) + " test_dice: " + str(test_dice))
                    
                    test_loss_list.append(test_loss)
                    test_acc_list.append(test_acc)
                    test_dice_list.append(test_dice)
                    test_writer.add_summary(summary, counter)
                    
        # Save in the end
        self.save(counter)
        
        # Collect train metrics
        train_metrics = {'loss':train_loss_list, 'acc':train_acc_list, 'dice':train_dice_list}
        test_metrics = {'loss':test_loss_list, 'acc':test_acc_list, 'dice':test_dice_list}
        
        return train_metrics,test_metrics
       
    def deploy(self, input_path, output_path):
        # Step 1
        if not self.load()[0]:
            raise Exception("No model is found, please train first") 
        
        # Apply this to all subjects including the training cases
        # Read from files.log and pick the testing cases for analysis
        all_paths = []
        # for dirpath, dirnames, files in os.walk(input_path):
        #     if os.path.basename(dirpath)[0:7] == 'Brats17':
        #         all_paths.append(dirpath)
        subject_dirs = next(os.walk(input_path))[1]
        for d in subject_dirs: 
            all_paths.append(os.path.join(input_path,d))
                
        for path in all_paths:
            image = read_image(path, is_training=False)
            locations, padding = generate_test_locations(self.patch_size, self.patch_stride, image.shape[:-1])
            pad_image = np.pad(image, padding + ((0, 0),), 'constant')
            pad_result = np.zeros((pad_image.shape[:-1] + (self.nclass,)), dtype=np.float32)
            pad_add = np.zeros((pad_image.shape[:-1]), dtype=np.float32)
            for x in locations[0]:
                for y in locations[1]:
                    for z in locations[2]:
                        patch = pad_image[int(x - self.patch_size[0] / 2) : int(x + self.patch_size[0] / 2),
                                          int(y - self.patch_size[1] / 2) : int(y + self.patch_size[1] / 2),
                                          int(z - self.patch_size[2] / 2) : int(z + self.patch_size[2] / 2), :]
                        
                        patch = np.expand_dims(patch, axis=0)
                        
                        probs = self.sess.run(self.probs, feed_dict = { self.images: patch,
                                                                        self.is_training: True,
                                                                        self.keep_prob: 1 })
                        pad_result[int(x - self.patch_size[0] / 2) : int(x + self.patch_size[0] / 2),
                                   int(y - self.patch_size[1] / 2) : int(y + self.patch_size[1] / 2),
                                   int(z - self.patch_size[2] / 2) : int(z + self.patch_size[2] / 2), :] += probs[0]
                        pad_add[int(x - self.patch_size[0] / 2) : int(x + self.patch_size[0] / 2),
                                int(y - self.patch_size[1] / 2) : int(y + self.patch_size[1] / 2),
                                int(z - self.patch_size[2] / 2) : int(z + self.patch_size[2] / 2)] += 1
            pad_result = pad_result / np.tile(np.expand_dims(pad_add, axis=3), (1, 1, 1, pad_result.shape[-1]))
            result = pad_result[padding[0][0] : padding[0][0] + image.shape[0],
                                padding[1][0] : padding[1][0] + image.shape[1],
                                padding[2][0] : padding[2][0] + image.shape[2], :]
            print(path)
            np.save(os.path.join(output_path, os.path.basename(path) + '_probs'), result)
            
    def estimate_mean_std(self, training_orders):
        means = []
        stds = []
        # Strictly speaking, this is not the correct way to estimate std since the mean 
        # used in each image is not the global mean but the mean of the image, this would
        # cause an over-estimation of the std.
        # The correct way may need much more memory, and more importantly, it probably does not matter...
        for order in training_orders:
            patch, _ = read_patch(os.path.join(self.training_paths[order[0]], str(order[1])),self.nclass)
            means.append(np.mean(patch, axis=(0, 1, 2)))
            stds.append(np.std(patch, axis=(0, 1, 2)))
        return np.mean(np.asarray(means, dtype=np.float32), axis=0), np.mean(np.asarray(stds, dtype=np.float32), axis=0)
    
    @property
    def model_dir(self):
        return 'unet3d_layer{}_{}'.format(self.layers, self.loss_type)
    
    def save(self, step):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'unet3d'), global_step=step)
        
    def load(self):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer('(\d+)(?!.*\d)', ckpt_name)).group(0))
            return True, counter
        else:
            print("Failed to find a checkpoint")
            return False, 0

#