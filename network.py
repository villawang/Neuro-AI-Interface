import tensorflow as tf
import os
import pdb


def shallow_networks():
    images = tf.placeholder(tf.float32,(None,128,128,3), 'input')
    rate = 0
    w1 = tf.get_variable('w1', shape=[5, 5, 3, 128],
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,
                    regularizer=tf.contrib.layers.l2_regularizer(1.))
    conv_layer1 = tf.nn.conv2d(images, w1, strides=[1, 1, 1, 1], padding='VALID')
    conv_layer1_pool = tf.layers.max_pooling2d(conv_layer1, pool_size=[5, 5], strides = [5, 5],
                                               padding='VALID')
    conv_layer1_act =  tf.nn.relu(conv_layer1_pool)


    w2 = tf.get_variable('w2', shape=[3, 3, 128, 64],
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,
                    regularizer=tf.contrib.layers.l2_regularizer(1.))
    conv_layer2 = tf.nn.conv2d(conv_layer1_act, w2, strides=[1, 1, 1, 1], padding='VALID')
    conv_layer2_pool = tf.layers.max_pooling2d(conv_layer2, pool_size=[3, 3], strides = [5, 5],
                                               padding='VALID')
    conv_layer2_act =  tf.nn.relu(conv_layer2_pool)


    image_feature = tf.contrib.layers.flatten(conv_layer2_act)
    w3 = tf.get_variable('w3', shape=[1024, 512],
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,
                        regularizer=tf.contrib.layers.l2_regularizer(1.))
    fc2 = tf.matmul(image_feature, w3)
    fc2_relu = tf.nn.relu(fc2)



    w4 = tf.get_variable('w4', shape=[512, 50],
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,
                        regularizer=tf.contrib.layers.l2_regularizer(1.))
    eeg_logits = tf.matmul(fc2_relu, w4)




    # EEG to neural score
    w5 = tf.get_variable('w5', shape=[50, 1],
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,
                        regularizer=tf.contrib.layers.l2_regularizer(1.))
    nr_logits = tf.matmul(eeg_logits, w5)
    return nr_logits, eeg_logits, [w1, w2, w3, w4], [w5]






# mobilenet v2 pretrained
def mobilenet_v2(config):
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    MODEL_DIR = './model_ckpt/mobilenet_v2_1.4_224'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'mobilenet_v2_1.4_224_frozen.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    # Works with an arbitrary minibatch size.
    with tf.Session(config=config) as sess:
#         sess.graph.get_tensor_by_name('input:0').__dict__['_shape_val'] = tf.TensorShape([None, 128,128,3])
        image_feature_layer = sess.graph.get_tensor_by_name('MobilenetV2/Logits/AvgPool:0')
        image_feature_layer.set_shape([None, 1,1,1792])

#         ops = image_feature.graph.get_operations()

#         for op_idx, op in enumerate(ops):
#             for o in op.outputs:
#                 shape = o.get_shape()
#                 shape = [s.value for s in shape]
#                 new_shape = []
#                 for j, s in enumerate(shape):
#                     if s == 1 and j == 0:
#                         new_shape.append(None)
#                     else:
#                         new_shape.append(s)
#                 o.__dict__['_shape_val'] = tf.TensorShape(new_shape)

        rate = 0
        image_feature = tf.layers.flatten(image_feature_layer)
        w1 = tf.get_variable('w1', shape=[1792, 896],
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(1.))
        image_EEG_layer1 = tf.matmul(image_feature, w1)
        image_EEG_layer1_activation = tf.nn.relu(image_EEG_layer1)
        image_EEG_layer1_activation_drop = tf.layers.dropout(image_EEG_layer1_activation, rate)

        w2 = tf.get_variable('w2', shape=[896, 448],
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(1.))
        image_EEG_layer2 = tf.matmul(image_EEG_layer1_activation_drop, w2)
        image_EEG_layer2_activation = tf.nn.relu(image_EEG_layer2)
        image_EEG_layer2_activation_drop = tf.layers.dropout(image_EEG_layer2_activation, rate)


        w3 = tf.get_variable('w3', shape=[448, 50],
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(1.))
        eeg_layer = tf.matmul(image_EEG_layer2_activation_drop, w3)

#         eeg_layer_activation = tf.nn.relu(eeg_layer)
#         eeg_layer_dropout = tf.layers.dropout(eeg_layer, rate)
        w4 = tf.get_variable('w4', shape=[50, 1],
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(1.))
        out = tf.matmul(eeg_layer, w4)
        print('Mobilenet_v2 graph loaded!')
        return out, eeg_layer, [w1, w2, w3], [w4]





# inception v3 pretrained
def inception_v3(config):
    MODEL_DIR = './model_ckpt/inception_v3'
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    # Works with an arbitrary minibatch size.
    with tf.Session(config=config) as sess:
#         sess.graph.get_tensor_by_name('ExpandDims:0').__dict__['_shape_val'] = tf.TensorShape([None, 128,128,3])
        image_feature_layer = sess.graph.get_tensor_by_name('pool_3:0')
        ops = image_feature_layer.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)

        rate = 0

        # image translation to EEG
        image_feature_flatten = tf.contrib.layers.flatten(image_feature_layer)

        w1 = tf.get_variable('w1', shape=[2048, 1024],
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(1.))
        image_EEG_layer1 = tf.matmul(image_feature_flatten, w1)
        image_EEG_layer1_activation = tf.nn.relu(image_EEG_layer1)
        image_EEG_layer1_activation_drop = tf.layers.dropout(image_EEG_layer1_activation, rate)

        w2 = tf.get_variable('w2', shape=[1024, 512],
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(1.))
        image_EEG_layer2 = tf.matmul(image_EEG_layer1_activation_drop, w2)
        image_EEG_layer2_activation = tf.nn.relu(image_EEG_layer2)
        image_EEG_layer2_activation_drop = tf.layers.dropout(image_EEG_layer2_activation, rate)


        w3 = tf.get_variable('w3', shape=[512, 50],
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(1.))
        eeg_logits = tf.matmul(image_EEG_layer2_activation_drop, w3)
#         eeg_logits_activation = tf.nn.relu(eeg_logits)


        # EEG to neural score
        w4 = tf.get_variable('w4', shape=[50, 1],
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(1.))
        nr_logits = tf.matmul(eeg_logits, w4)

        # dirty way to overcome the shape problem in tensorflow
        eeg_logits.__dict__['_shape_val'] = tf.TensorShape([None, 50])
        nr_logits.__dict__['_shape_val'] = tf.TensorShape([None, 1])

        print('Inception graph loaded!')
        return nr_logits, eeg_logits, [w1, w2, w3], [w4]
