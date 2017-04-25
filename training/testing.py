import tensorflow as tf
import pandas as pd
import numpy as np

from architecture import modeltype
import skimage.io
import skimage.transform
import ipdb

import numpy as np

def load_image( path ):
    try:
        img = skimage.io.imread( path ).astype( float )
    except:
        return None

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    img /= 255.

    short_edge = min( img.shape[:2] )
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = skimage.transform.resize( crop_img, [224,224] )
    return resized_img


import skimage.io
import matplotlib.pyplot as plt

import os
import ipdb

testset_path = 'data/caltech/test.pickle'
label_dict_path = 'data/caltech/label_dict.pickle'

weight_path = 'data/caffe_layers_value.pickle'
model_path = 'models/caltech256/'

batch_size = 1

testset = pd.read_pickle( testset_path )[::-1][:20]
label_dict = pd.read_pickle( label_dict_path )
n_labels = len( label_dict )

images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels')

modeon = modeltype( weight_path, n_labels )
c1,c2,c3,c4,conv5, conv6, gap, output = modeon.inference( images_tf )
classmap = modeon.get_classmap( labels_tf, conv6 )

sess = tf.InteractiveSession()
saver = tf.train.Saver()

saver.restore( sess, model_path )

for start, end in zip(
    range( 0, len(testset)+batch_size, batch_size),
    range(batch_size, len(testset)+batch_size, batch_size)):

    current_data = testset[start:end]
    current_image_paths = current_data['image_path'].values
    current_images = np.array(map(lambda x: load_image(x), current_image_paths))

    good_index = np.array(map(lambda x: x is not None, current_images))

    current_data = current_data[good_index]
    current_image_paths = current_image_paths[good_index]
    current_images = np.stack(current_images[good_index])
    current_labels = current_data['label'].values
    current_label_names = current_data['label_name'].values

    conv6_val, output_val = sess.run(
            [conv6, output],
            feed_dict={
                images_tf: current_images
                })

    label_predictions = output_val.argmax( axis=1 )
    acc = (label_predictions == current_labels).sum()

    classmap_vals = sess.run(
            classmap,
            feed_dict={
                labels_tf: label_predictions,
                conv6: conv6_val
                })

    classmap_answer = sess.run(
            classmap,
            feed_dict={
                labels_tf: current_labels,
                conv6: conv6_val
                })

    classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_answer)

    for vis, ori,ori_path, l_name in zip(classmap_vis, current_images, current_image_paths, current_label_names):
        print l_name
        plt.imshow( ori )
        plt.imshow( vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest' )
        plt.show()

