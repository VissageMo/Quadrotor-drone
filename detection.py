import tensorflow as tf
import numpy as np
import time
import pickle
import tensorboard


p = []
batch_size = 8
learning_rate = 0.001
max_num = 50000
image_height = 480
image_width = 640
n_class = 2
hl, hs = 20, 15
wl, ws = 60, 45

X = tf.placeholder(tf.float32, [None, image_height, image_width, 3])
Y = tf.placeholder(tf.float32, [None, n_class])
keep_prob = tf.placeholder(tf.float32)


def create_data(batch_size, in_x, in_y, hl, hs, wl, ws):
    # b = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    # g = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    # r = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

    out_batch = np.random.randint(0, 255, (batch_size, 480, 640, 3), dtype=np.uint8)
    x = np.random.randint(0, 480, (1, batch_size))
    y = np.random.randint(0, 640, (1, batch_size))

    for i in range(batch_size):

        h1, h2 = max(x[0, i] - hl, 0), min(x[0, i] + hl, 480)
        w1, w2 = max(y[0, i] - wl, 0), min(y[0, i] + wl, 640)
        h3, h4 = max(x[0, i] - hs, 0), min(x[0, i] + hs, 480)
        w3, w4 = max(y[0, i] - ws, 0), min(y[0, i] + ws, 640)

        out_batch[i, h1:h2, w1:w2, :] = 255, 0, 0
        out_batch[i, h3:h4, w3:w4, 1:3] = 255, 255


def variable_with_weight_loss(shape, stddev, wl):
    # initialize weight
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    # create L2 regularization based on weight loss and save it in losses
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


# save the kernel and biases in p
def conv2(input_op, name, n_out, kh, kw, dh, dw, p=p, padding='SAME'):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        # kernel = variable_with_weight_loss(shape=[kh, kw, n_in, n_out], stddev=5e-2, wl=0.0)
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding=padding)
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


def fc(input_op, name, n_out, p=p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation


def mpool(input_op, name, kh, kw, dh, dw, padding='SAME'):
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1],
                          padding=padding, name=name)


def apool(input_op, name, kh, kw, dh, dw, padding='SAME'):
    return tf.nn.avg_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1],
                          padding=padding, name=name)


def alexnet(input_op):

    net = conv2(input_op, 'conv1', 64, 5, 5, 2, 2)
    net = mpool(net, 'mpool1', 3, 3, 2, 2)
    net = conv2(net, 'conv2', 128, 5, 5, 2, 2)
    net = mpool(net, 'mpool2', 3, 3, 2, 2)
    net = conv2(net, 'conv3', 256, 3, 3, 1, 1)
    net = conv2(net, 'conv4', 512, 3, 3, 1, 1)
    net = conv2(net, 'conv5', 256, 3, 3, 1, 1)
    net = mpool(net, 'mpool3', 2, 2, 1, 1, 'VALID')
    net = tf.reshape(net, [-1, 256])
    net = fc(net, 'fc1', 1024)
    net = fc(net, 'fc2', 1024)
    net = fc(net, 'fc3', 10)

    return net


def vgg_16(input_op):

    conv1_1 = conv2(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv2(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

    conv2_1 = conv2(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv2(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1,dw=1, p=p)
    pool2 = mpool(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    conv3_1 = conv2(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv2(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv2(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    conv4_1 = conv2(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv2(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv2(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    conv5_1 = conv2(pool4, name="conv5_1", kw=3, kh=3, n_out=512, dw=1, dh=1, p=p)
    conv5_2 = conv2(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv2(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool(conv5_3, name="pool5", kh=2, kw=2, dh=2, dw=2)

    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    fc6 = fc(resh1, name="fc6", n_out=10)

    return fc6


acc = 0
P = []
train_images, train_labels = load_cifar10_train(cifar_10_path)
Ys = np.zeros((len(train_labels), n_class))

for i in range(len(train_labels)):
    j = train_labels[i]
    Ys[i, j] = 1


prediction = alexnet(X)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver(max_to_keep=2)


if __name__ == '__main__':

    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(iter_size):
            for j in range(max_num // batch_size):

                batch_x = train_images[j * batch_size: j * batch_size + batch_size]
                batch_y = Ys[j * batch_size: j * batch_size + batch_size]

                _, cost= sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.})
                acc += sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.})

                if j % 30 == 0:
                    print('Iter %d , epoch %d' % (i, i*390+j))
                    print(cost)

            acc = acc / 390
            print('### Iter %d ###' % i)
            print('average acc: %f' % acc)
            acc = 0

            if i % 5 == 0:
                saver.save(sess, '/Users/mowenhao/PycharmProjects/py3.6/Model/vgg/vgg16')
                print('Successful Saved')



