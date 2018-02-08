import tensorflow as tf


def batch_norm_layer(inputs, phase_train, scope=None):
    return tf.cond(phase_train,
                   lambda: tf.contrib.layers.batch_norm(inputs, is_training=True, scale=True,
                                                        updates_collections=None, scope=scope),
                   lambda: tf.contrib.layers.batch_norm(inputs, is_training=False, scale=True,
                                                        updates_collections=None, scope=scope, reuse=True))


def do_conv(x, layer_name, kernel_size, depth, stride_size, padding='SAME', is_activation=True, activation_method=None, is_bn=True, config=None):
    shape = x.get_shape().as_list()
    if len(shape) == 3:
        shape.extend(1)
    with tf.variable_scope(layer_name):
        weight = tf.get_variable('weight', shape=[kernel_size, kernel_size, shape[-1], depth],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', shape=[depth], initializer=tf.constant_initializer(value=0.0))
        output = tf.nn.conv2d(x, weight, strides=[1, stride_size, stride_size, 1], padding=padding)

        if is_bn:
            output = batch_norm_layer(output, config['is_training'], scope='batch_norm_layer')
        else:
            output = tf.nn.bias_add(output, bias)
        if is_activation:
            output = activation_method(output)
    return output


def do_pooling(x, method, layer_name, kernel_size, stride_size, padding='SAME'):
    with tf.variable_scope(layer_name):
        if method == 'max':
            output = tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride_size, stride_size, 1],
                                    padding=padding)
        else:
            output = tf.nn.avg_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride_size, stride_size, 1],
                                    padding=padding)
    return output


def do_fc(x, layer_name, depth, is_activation=True, activation_method=None):
    shape = x.get_shape().as_list()
    if len(shape) == 3:
        shape.extend(1)
    with tf.variable_scope(layer_name):
        weight = tf.get_variable('weight', shape=[shape[-1], depth],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', shape=[depth], initializer=tf.constant_initializer(value=0.0))
        output = tf.matmul(x, weight)
        output = tf.nn.bias_add(output, bias)
        if is_activation:
            output = activation_method(output)
    return output


def inference(x_tensor, is_training):
    config = {}
    config['is_training'] = tf.convert_to_tensor(is_training,
                                                 dtype='bool',
                                                 name='is_training')
    activation_method = tf.nn.relu

    conv1 = do_conv(x_tensor, 'layer1-conv1', kernel_size=3, depth=16, stride_size=1,
                    activation_method=activation_method, config=config)
    conv2 = do_conv(conv1, 'layer2-conv2', kernel_size=3, depth=16, stride_size=1, activation_method=activation_method,
                    config=config)

    pooling1 = do_pooling(conv2, method='max', layer_name='layer3-pooling1', kernel_size=3, stride_size=2)

    conv3 = do_conv(pooling1, layer_name='layer4-conv3', kernel_size=3, depth=32, stride_size=1,
                    activation_method=activation_method, config=config)
    conv4 = do_conv(conv3, layer_name='layer5-conv4', kernel_size=3, depth=32, stride_size=1,
                    activation_method=activation_method, config=config)

    pooling2 = do_pooling(conv4, method='max', layer_name='layer6-pooling2', kernel_size=3, stride_size=2)

    conv5 = do_conv(pooling2, layer_name='layer7-conv5', kernel_size=3, stride_size=1, depth=64,
                    activation_method=activation_method, config=config)
    conv6 = do_conv(conv5, layer_name='layer8-conv6', kernel_size=3, stride_size=1, depth=64,
                    activation_method=activation_method, config=config)

    pooling3 = do_pooling(conv6, method='max', layer_name='layer9-pooling3', kernel_size=3, stride_size=2)

    shape = pooling3.get_shape().as_list()
    pooling3 = tf.reshape(pooling3, [-1, shape[2] * shape[3] * shape[1]])
    print 'After flatten, the shape of pooling3 is ', shape[2] * shape[3] * shape[1]
    fc1 = do_fc(pooling3, layer_name='layer10-fc1', depth=256, activation_method=activation_method)
    fc2 = do_fc(fc1, layer_name='layer11-fc2', depth=128, activation_method=activation_method)
    fc3 = do_fc(fc2, layer_name='layer12-fc3', depth=4, activation_method=activation_method)
    return fc3

if __name__ == '__main__':
    input_tensor = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='x_input')
    output = inference(input_tensor, is_training=True)
    print output