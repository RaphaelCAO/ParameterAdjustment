# -*- coding:utf-8 -*-
import functools
import sys, cv2
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


sys.path.append('/home/raphael/PycharmProjects/bl/models/research/')
sys.path.append('/home/raphael/PycharmProjects/bl/models/research/slim')

from object_detection.builders import model_builder
from object_detection.utils import config_util

pipeline_config_path = "/home/raphael/PycharmProjects/bl/model_config/new_model/pb/pipeline.config"
IMAGE_SIZE = 300


def build_mdodel():
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']

    model_fn = functools.partial(model_builder.build, model_config=model_config, is_training=False)

    # 建立graph,流程参考evaluator.py中的_extract_predictions_and_losses(),主要调用ssd_mata_arch.py中的函数
    model = model_fn()
    return model


if __name__ == '__main__':
    model = build_mdodel()
    image = cv2.imread('/home/raphael/caoqi_0_ver1.jpg')
    image_height, image_width, _ = image.shape

    x = tf.placeholder(dtype=tf.float32, shape=[1, image_height, image_width, 3])
    preprocessed_image,  true_image_shapes = model.preprocess(tf.to_float(x))
    prediction_dict = model.predict(preprocessed_image, true_image_shapes)
    detections, all_scores_no_bg = model.postprocess(prediction_dict, true_image_shapes)

    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = np.expand_dims(image_np, axis=0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        detections = sess.run(detections, feed_dict={x: original_image})
        print(detections)



