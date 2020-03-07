# coding=utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob
import TensorflowUtils as utils
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary, unary_from_labels

# import skimage.io as io
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

# 参数设置
# 将data_url改成需要的图片
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-6", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_string("pic_final_dir", "Final_Pic/", "path to pic_final directory")
tf.flags.DEFINE_bool('debug', "True", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "test", "Mode train/ test/ visualize")
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
#将类别数改为4
MAX_ITERATION = 20000        # 迭代次数
NUM_OF_CLASSESS = 256                # 类别数 151
IMAGE_SIZE = 400                    # 图片大小 224
fine_tuning = False

# VGG网络部分，weights是权重集合， image是预测图像的向量
def vgg_net(weights, image):
    # VGG网络前五大部分
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image     # 预测图像
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")     # conv1_1_w
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")       # conv1_1_b
            current = utils.conv2d_basic(current, kernels, bias)        # 前向传播结果 current
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)    # relu1_1
            if FLAGS.debug:     # 是否开启debug模式 true / false
                utils.add_activation_summary(current)       # 画图
        elif kind == 'pool':
            # vgg 的前5层的stride都是2，也就是前5层的size依次减小1倍
            # 这里处理了前4层的stride，用的是平均池化
            # 第5层的pool在下文的外部处理了，用的是最大池化
            # pool1 size缩小2倍
            # pool2 size缩小4倍
            # pool3 size缩小8倍
            # pool4 size缩小16倍
            current = utils.avg_pool_2x2(current)
        net[name] = current     # 每层前向传播结果放在net中， 是一个字典

    return net


# 预测流程，image是输入图像，keep_prob dropout比例
def inference(image, keep_prob):
    """
    Semantic segmentation network definition    # 语义分割网络定义
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    # 获取预训练网络VGG
    print("setting up vgg initialized conv layers ...")
    # model_dir Model_zoo/
    # MODEL_URL 下载VGG19网址
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)       # 返回VGG19模型中内容

    mean = model_data['normalization'][0][0][0]                         # 获得图像均值
    mean_pixel = np.mean(mean, axis=(0, 1))                             # RGB

    weights = np.squeeze(model_data['layers'])                          # 压缩VGG网络中参数，把维度是1的维度去掉 剩下的就是权重

    processed_image = utils.process_image(image, mean_pixel)            # 图像减均值

    with tf.variable_scope("inference"):                                # 命名作用域 是inference
        image_net = vgg_net(weights, processed_image)                   # 传入权重参数和预测图像，获得所有层输出结果
        # conv_final_layer = image_net["conv5_3"]                         # 获得输出结果
        conv_final_layer = image_net["relu4_4"]
        w5_0 = utils.weight_variable([3, 3, 512, 512], name="W5_0")     #取消pool4降采样操作，改成3*3/s1
        b5_0 = utils.bias_variable([512], name="b5_0")
        conv5_0 = utils.conv2d_strided(conv_final_layer, w5_0, b5_0)

        w5_1 = utils.weight_variable([3, 3, 512, 512], name="W5_1")
        b5_1 = utils.bias_variable([512], name="b5_1")
        conv5_1 = utils.conv2d_atrous_2(conv5_0, w5_1, b5_1)
        w5_2 = utils.weight_variable([3, 3, 512, 512], name="W5_2")      #将第五层的conv5_1,2,3改成2-空洞卷积
        b5_2 = utils.bias_variable([512], name="b5_2")
        conv5_2 = utils.conv2d_atrous_2(conv5_1, w5_2, b5_2)
        w5_3 = utils.weight_variable([3, 3, 512, 512], name="W5_3")
        b5_3 = utils.bias_variable([512], name="b5_3")
        conv5_3 = utils.conv2d_atrous_2(conv5_2, w5_3, b5_3)
        # pool5 = utils.max_pool_2x2(conv_final_layer)                    # /32 缩小32倍

        # W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")        # 初始化第6层的w b
        # b6 = utils.bias_variable([4096], name="b6")
        # conv6 = utils.conv2d_basic(pool5, W6, b6)
        # relu6 = tf.nn.relu(conv6, name="relu6")
        # if FLAGS.debug:
        #     utils.add_activation_summary(relu6)
        # relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        w6_0 = utils.weight_variable([3, 3, 512, 4096], name="W6_0")  # 取消pool5降采样操作，改成3*3/s1
        b6_0 = utils.bias_variable([4096], name="b6_0")
        conv6_0 = utils.conv2d_strided(conv5_3, w6_0, b6_0)

        w6 = utils.weight_variable([3, 3, 4096, 4096], name="W7")
        b6=utils.bias_variable([4096],name="b6")                          #第6层为4-空洞卷积
        conv6=utils.conv2d_atrous_4(conv6_0,w6,b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        # W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")       # 第7层卷积层
        # b7 = utils.bias_variable([4096], name="b7")
        # conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        # relu7 = tf.nn.relu(conv7, name="relu7")
        # if FLAGS.debug:
        #     utils.add_activation_summary(relu7)
        # relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        w7 = utils.weight_variable([1, 1, 4096, 4096], name="w7")
        b7 = utils.bias_variable([4096], name="b7")                       #第7层为4—空洞卷积
        conv7 = utils.conv2d_atrous_4(relu_dropout6, w7, b7)
        relu7 = tf.nn.relu(conv7, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        # W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        # b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        # conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)               # 第8层卷积层 分类151类
        # # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")
        w8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")  # 第8层为4—空洞卷积
        conv8 = utils.conv2d_atrous_4(relu_dropout7, w8, b8)
        # conv8 = utils.max_pool_2x2(conv8)
        print(conv8.shape)
        # now to upscale to actual image size
        # deconv_shape1 = image_net["pool4"].get_shape()                  # 将pool4 1/16结果尺寸拿出来 做融合 [b,h,w,c]
        # # 定义反卷积层的 W，B [H, W, OUTC, INC]  输出个数为pool4层通道个数，输入为conv8通道个数
        # # 扩大两倍  所以stride = 2  kernel_size = 4
        # W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        # b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        # # 输入为conv8特征图，使得其特征图大小扩大两倍，并且特征图个数变为pool4的通道数
        # conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        # fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")     # 进行融合 逐像素相加

        deconv_shape1 = image_net["pool4"].get_shape()                  # 将pool4 1/16结果尺寸拿出来 做融合 [b,h,w,c]
        # 定义反卷积层的 W，B [H, W, OUTC, INC]  输出个数为pool4层通道个数，输入为conv8通道个数
        # 扩大两倍  所以stride = 2  kernel_size = 4
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        # 输入为conv8特征图，使得其特征图大小扩大两倍，并且特征图个数变为pool4的通道数
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")     # 进行融合 逐像素相加

        # 获得pool3尺寸 是原图大小的1/8
        deconv_shape2 = image_net["pool3"].get_shape()
        # 输出通道数为pool3通道数，  输入通道数为pool4通道数
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        # 将上一层融合结果fuse_1在扩大两倍，输出尺寸和pool3相同
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        # 融合操作deconv(fuse_1) + pool3
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)     # 获得原始图像大小
        # 堆叠列表，反卷积输出尺寸，[b，原图H，原图W，类别个数]
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        # 建立反卷积w[8倍扩大需要ks=16, 输出通道数为类别个数， 输入通道数pool3通道数]
        W_t3 = utils.weight_variable([16, 16, deconv_shape2[3].value, NUM_OF_CLASSESS], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        # 反卷积，fuse_2反卷积，输出尺寸为 [b，原图H，原图W，类别个数]
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        # deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        # # 建立反卷积w[8倍扩大需要ks=16, 输出通道数为类别个数， 输入通道数pool3通道数]
        # W_t1 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, NUM_OF_CLASSESS], name="W_t1")  ##反卷积生成原图大小
        # b_t1 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t1")
        # # 反卷积，fuse_2反卷积，输出尺寸为 [b，原图H，原图W，类别个数]
        # conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=deconv_shape3, stride=8)

        # 目前conv_t3的形式为size为和原始图像相同的size，通道数与分类数相同
        # 这句我的理解是对于每个像素位置，根据第3维度（通道数）通过argmax能计算出这个像素点属于哪个分类
        # 也就是对于每个像素而言，NUM_OF_CLASSESS个通道中哪个数值最大，这个像素就属于哪个分类
        # 每个像素点有21个值，哪个值最大就属于那一类
        # 返回一张图，每一个点对于其来别信息shape=[b,h,w]
        # annotation_pred = tf.argmax(conv_t1, dimension=3, name="prediction")
        annotation_pred = tf.argmax(conv_t3, axis=3, name="prediction")
    # 从第三维度扩展 形成[b,h,w,c] 其中c=1, conv_t3最后具有21深度的特征图
    # return tf.expand_dims(annotation_pred, dim=3), conv_t1

    return tf.expand_dims(annotation_pred, axis=3), conv_t3


def train(loss_val, var_list):
    """

    :param loss_val:  损失函数
    :param var_list:  需要优化的值
    :return:
    """
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)     # 返回迭代梯度


def main(argv=None):
    # dropout保留率
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    # 图像占坑
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    # 标签占坑
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    # 预测一个batch图像  获得预测图[b,h,w,c=1]  结果特征图[b,h,w,c=151]
    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    # 空间交叉熵损失函数[b,h,w,c=151]  和labels[b,h,w]    每一张图分别对比
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, axis=[3]),
                                                                          name="entropy")))
    tf.summary.scalar("entropy", loss)

    raw_output_up = tf.image.resize_bilinear(logits, tf.shape(image)[0:2,])
    raw_output_up_squeeze =tf.squeeze(raw_output_up,axis=0)
    raw_output_up_squeeze=tf.nn.softmax(raw_output_up_squeeze,)
    probabilities = tf.nn.softmax(logits)
    
    # 返回需要训练的变量列表
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)

    # 传入损失函数和需要训练的变量列表
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    # 生成绘图数据
    summary_op = tf.summary.merge_all()
    
    if FLAGS.mode != "test":
        print("Setting up image reader...")
        # data_dir = Data_zoo/MIT_SceneParsing/
        # training: [{image: 图片全路径， annotation:标签全路径， filename:图片名字}] [{}][{}]
        test_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
        print(len(test_records))   # 长度
        # print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    
    if FLAGS.mode == 'train':
        # 读取图片 产生类对象 其中包含所有图片信息
        train_dataset_reader = dataset.BatchDatset(test_records, image_options)

    if FLAGS.mode != "test":
        validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)
    sess.run(tf.global_variables_initializer())

    if fine_tuning:
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)    # 训练断点回复
        if ckpt and ckpt.model_checkpoint_path:                 # 如果存在checkpoint文件 则恢复sess
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")

    if FLAGS.mode == "train":
        for itr in range(MAX_ITERATION):
            # 读取下一batch
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            # 迭代优化需要训练的变量
            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                # 迭代10次打印显示
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)    #调用train_writer的add_summary方法将训练过程以及训练步数保存  

            if itr % 500 == 0:
                # 迭代500 次验证
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                # 保存模型
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        # 可视化
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        # pred_annotation预测结果图
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)

    elif FLAGS.mode == "test":
        file_list = []
        # ./ number_segment_2/img.jpg
        # 加入文件列表  包含所有图片文件全路径+文件名字  如 Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/training/hi.jpg
        file_glob=os.path.join(FLAGS.data_dir+'ADEChallengeData2016/', "images/", 'validation/', '*.' + 'jpg')
        file_list.extend(glob.glob(file_glob))
        
        print('Start Transport')

        for f in file_list:
            filename = os.path.splitext(f.split("/")[-1])[0]
            filename = os.path.splitext(filename.split("_")[-1])[0]
            image_orignal= misc.imread(f)
            # resize_image = misc.imresize(image_orignal,
            #                              [224, 224], interp='nearest')
            image_final=np.array([image_orignal])
            pred ,probabilities_np= sess.run([pred_annotation, probabilities], feed_dict={image: image_final, keep_probability: 1.0})
            image_pred=np.squeeze(pred)
            image1 = image_orignal
            # softmax = probabilities_np.squeeze()
            softmax = probabilities_np.transpose((2, 0, 1, 3))

            # softmax_to_unary函数的输入数据为概率值的负对数
            unary = softmax_to_unary(softmax)

            # 输入数据应该是 C-continious -- 这里采用 Cython 封装器
            unary = np.ascontiguousarray(unary)

            d = dcrf.DenseCRF(image1.shape[0] * image1.shape[1], 256)
            
            n_labels = len(set(image_pred.flat))
            unary = unary_from_labels(image_pred, n_labels, gt_prob=0.7, zero_unsure=0)
            d.setUnaryEnergy(unary)

            # 对空间独立的小分割区域进行潜在地惩罚 - 促使生成更多连续的分割区域.
            feats = create_pairwise_gaussian(sdims=(5, 5), shape=image1.shape[:2])

            d.addPairwiseEnergy(feats, compat=3,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

            # 创建与颜色相关的特征
            # 因为 CNN 的分割结果太粗糙，使用局部的颜色特征来进一步提升分割结果.
            feats = create_pairwise_bilateral(sdims=(100, 100), schan=(20, 20, 20),
                                              img=image1, chdim=2)

            d.addPairwiseEnergy(feats, compat=10,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)
            Q = d.inference(5)

            res = np.argmax(Q, axis=0).reshape((image1.shape[0], image1.shape[1]))
            utils.save_image(res.astype(np.uint8), FLAGS.pic_final_dir, name='img'+'_'+filename )
            # cmap = plt.get_cmap('bwr')
            #
            # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            # ax1.imshow(res, vmax=1.5, vmin=-0.4, cmap=cmap)
            # ax1.set_title('Segmentation with CRF post-processing')
            # probability_graph = ax2.imshow(np.dstack((train_annotation,) * 3) * 100)
            # ax2.set_title('Ground-Truth Annotation')
            # plt.show()
            # print(image_pred.shape)
            # pred_400 = misc.imresize(image_pred,
            #                              [400, 400], interp='nearest')
            # utils.save_image(image_pred.astype(np.uint8), FLAGS.pic_dir, name='img'+'_'+filename )



if __name__ == "__main__":
    tf.app.run()
