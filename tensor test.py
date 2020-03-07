# shape of pool1 is (32, 56, 56, 64)
# shape of blocks_dense is (32, 3, 3, 2048)
# shape of postnorm is (32, 3, 3, 2048)
# shape of pool5 is (32, 1, 1, 2048)
# shape of logits is (32, 1, 1, 256)
# shape of dict is (32, 1, 1, 256)

# (?, 14, 14, 512)

import TensorflowUtils as utils
import tensorflow as tf

NUM_OF_CLASSESS = 256

# the shape of conv_final_layer is (?, 14, 14, 512)
pool5 = tf.random_uniform((32, 1, 1, 2048))
print("the shape of pool5 is %s" %pool5.shape)

# print("the shape of pool5 is %s" %pool5.shape)
# # the shape of relu_dropout6 is (?, 7, 7, 4096)     
# W6 = utils.weight_variable([1, 1, 4096, 2048], name="W6")        # 初始化第6层的w b
# b6 = utils.bias_variable([4096], name="b6")
# conv6 = utils.conv2d_transpose_strided(pool5, W6, b6, output_shape=[-1, 7, 7, 4096], stride = 8)# (24, 14, 14, 256)
# relu6 = tf.nn.relu(conv6, name="relu6")							# (24, 7, 7, 4096)
# relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
# print("the shape of relu_dropout6 is %s" %relu_dropout6.shape)

pool5 = utils.max_unpool(pool5,[7,7])
print("the shape of pool5 is %s" %pool5.shape)

keep_prob = 1.0
# the shape of relu_dropout6 is (?, 7, 7, 4096)
W6 = utils.weight_variable([1, 1, 2048, 4096], name="W6")        # 初始化第6层的w b
b6 = utils.bias_variable([4096], name="b6")
conv6 = utils.conv2d_basic(pool5, W6, b6)
relu6 = tf.nn.relu(conv6, name="relu6")							# (24, 7, 7, 4096)
relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
print("the shape of relu_dropout6 is %s" %relu_dropout6.shape)

# the shape of relu_dropout7 is (?, 7, 7, 4096)
W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")       # 第7层卷积层
b7 = utils.bias_variable([4096], name="b7")
conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
relu7 = tf.nn.relu(conv7, name="relu7")
relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
print("the shape of relu_dropout7 is %s" %relu_dropout7.shape)  #(?, 7, 7, 4096)

# the shape of conv8 is (?, 7, 7, 256)
W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)               # 第8层卷积层 分类151类
# annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")
print("the shape of conv8 is %s" %conv8.shape)					#(?, 7, 7, 256)

# the shape of fuse_1 is (?, 14, 14, 512)
W_t1 = utils.weight_variable([4, 4, 512, NUM_OF_CLASSESS], name="W_t1")
b_t1 = utils.bias_variable([512], name="b_t1")
conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1)		# (24, 14, 14, 256)
print("the shape of conv_t1 is %s" %conv_t1.shape)

# 获得pool3尺寸 是原图大小的1/8
# 输出通道数为pool3通道数，  输入通道数为pool4通道数
# the shape of fuse_2 is (?, 28, 28, 256)
W_t2 = utils.weight_variable([4, 4, NUM_OF_CLASSESS, 512], name="W_t2")
b_t2 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t2")
# 将上一层融合结果fuse_1在扩大两倍，输出尺寸和pool3相同
conv_t2 = utils.conv2d_transpose_strided(conv_t1, W_t2, b_t2)
# 融合操作deconv(fuse_1) + pool3
print("the shape of conv_t2 is %s" %conv_t2.shape)

# 堆叠列表，反卷积输出尺寸，[b，原图H，原图W，类别个数]
# the shape of conv_t3 is (?, ?, ?, 256)
deconv_shape3 = tf.stack([-1, 224, 224, NUM_OF_CLASSESS])
# 建立反卷积w[8倍扩大需要ks=16, 输出通道数为类别个数， 输入通道数pool3通道数]
W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, NUM_OF_CLASSESS], name="W_t3")
b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
# 反卷积，fuse_2反卷积，输出尺寸为 [b，原图H，原图W，类别个数]
# conv_t3 = utils.conv2d_transpose_strided(conv_t2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
conv_t3 = utils.conv2d_transpose_strided8(conv_t2, W_t3, b_t3, stride = 8)
print("the shape of conv_t3 is %s" %conv_t3.shape)

# 从第三维度扩展 形成[b,h,w,c] 其中c=1, conv_t3最后具有21深度的特征图
annotation_pred = tf.argmax(conv_t3, axis=3, name="prediction")
