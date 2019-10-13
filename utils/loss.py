import numpy as np
import tensorflow as tf


def deep_metric_loss(y_true, y_pred):
    loss = tf.py_func(deep_metric_loss_py, [y_true, y_pred], tf.float32)

    return loss



def distance_metric(vec1, vec2):

    # euclidean distance
    distance = np.square(vec1-vec2)
    distance = np.sum(distance)

    return distance


def deep_metric_loss_py(y_true, y_pred):

    pos_embedding = y_pred[y_true>0]
    neg_embedding = y_pred[y_true==0]

    pos_distance = 0
    count = 0
    for i in range(len(pos_embedding)):
        for k in range(i + 1, len(pos_embedding)):
            pos_distance += distance_metric(pos_embedding[i], pos_embedding[k])
            count+=1

    pos_distance /= ((len(pos_embedding)+1)*len(pos_embedding)/2)

    loss = 0
    count = 0

    for i in range(len(pos_embedding)):
        for k in range(len(neg_embedding)):
            temp = distance_metric(pos_embedding[i], neg_embedding[k])

            if temp <= pos_distance:
                loss += abs(temp - pos_distance)
                count+=1

    if count==0:
        return 0

    return loss/count
