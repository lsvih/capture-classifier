import tensorflow as tf


class ConvNet():
    def __init__(self):
        self.channel = 3  # RGB
        self.classes = 4  # splice into 4 classes
        self.size = 200  # images size 200 x 200 px
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[None, self.size, self.size, self.channel], name='images')
        self.labels = tf.placeholder(
            dtype=tf.int64, shape=[None], name='labels')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        model = self.model()
        # 目标函数
        self.objective = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=self.labels))
        tf.add_to_collection('loss', self.objective)
        self.loss = tf.add_n(tf.get_collection('loss'))
        # 用 Adam 优化器最小化 loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        # 准确率
        correct_prediction = tf.equal(self.labels, tf.argmax(model, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        # Session
        self.sess = None

    def model(self):
        # shape: size, size, channel
        conv_layer1 = tf.layers.conv2d(
            inputs=self.images,
            kernel_size=3,
            filters=64,
            strides=1,
            padding="SAME",
            activation=tf.nn.relu,
            use_bias=True,
            name="conv1"
        )

        pool_layer1 = tf.layers.max_pooling2d(
            inputs=conv_layer1,
            pool_size=2,
            strides=2,
            padding="SAME",
            name="pool1"
        )
        # shape: size/2, size/2, 64

        conv_layer2 = tf.layers.conv2d(
            inputs=pool_layer1,
            kernel_size=3,
            filters=128,
            strides=1,
            padding="SAME",
            activation=tf.nn.relu,
            use_bias=True,
            name="conv2"
        )

        pool_layer2 = tf.layers.max_pooling2d(
            inputs=conv_layer2,
            pool_size=2,
            strides=2,
            padding="SAME",
            name="pool2"
        )
        # shape: size/4, size/4, 128
        conv_layer3 = tf.layers.conv2d(
            inputs=pool_layer2,
            kernel_size=3,
            filters=256,
            strides=1,
            padding="SAME",
            activation=tf.nn.relu,
            use_bias=True,
            name="conv3"
        )
        pool_layer3 = tf.layers.max_pooling2d(
            inputs=conv_layer3,
            pool_size=2,
            strides=2,
            padding="SAME",
            name="pool2"
        )
        # shape: size/8, size/8, 256
        conv_output = tf.reshape(
            tensor=pool_layer3,
            shape=[-1, int(self.size / 8) ** 2 * 256]
        )
        # shape: size/8 * size/8 * 256
        dense_layer1 = tf.layers.dense(
            inputs=conv_output,
            units=1024,
            activation=tf.nn.relu,
            name="dense1"
        )
        dropout_1 = tf.layers.dropout(
            inputs=dense_layer1,
            rate=0.5,
            name="dropout1"
        )
        output = tf.layers.dense(
            inputs=dropout_1,
            units=self.classes,
            activation=None,
            name="dense2"
        )
        return output

    def train(self, dataset, n_epoch=5, batch_size=128):
        # 构建会话
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.48)  # 等机房修好了就能在服务器上跑了QAQ
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        # 模型训练
        for epoch in range(0, n_epoch + 1):
            train_images, train_labels = dataset["train_images"], dataset["train_labels"]  # 数据增强
            valid_images, valid_labels = dataset["valid_images"], dataset["valid_labels"]

            train_loss = 0.0
            # 按照 batch size 批量训练
            for i in range(0, len(train_images), batch_size):
                images, labels = train_images[i: i + batch_size], train_labels[i: i + batch_size]
                [_, loss, _] = self.sess.run(
                    fetches=[self.optimizer, self.loss],
                    feed_dict={self.images: images,
                               self.labels: labels,
                               self.keep_prob: 0.5})
                train_loss += loss * images.shape[0]
            train_loss = 1.0 * train_loss / len(train_images)

            # 计算验证集的 loss 和准确率
            valid_accuracy, valid_loss = 0.0, 0.0
            for i in range(0, len(valid_images), batch_size):
                images, labels = valid_images[i: i + batch_size], valid_labels[i: i + batch_size]
                [avg_accuracy, loss] = self.sess.run(
                    fetches=[self.accuracy, self.loss],
                    feed_dict={self.images: images,
                               self.labels: labels,
                               self.keep_prob: 1.0})
                valid_accuracy += avg_accuracy * images.shape[0]
                valid_loss += loss * images.shape[0]
            valid_accuracy = 1.0 * valid_accuracy / len(valid_images)
            valid_loss = 1.0 * valid_loss / len(valid_images)

            print('epoch{%d}, 训练集loss: %.6f, '
                  '验证准确率: %.6f, 验证集loss: %.6f' % (
                      epoch, train_loss, valid_accuracy, valid_loss))
        self.sess.close()
