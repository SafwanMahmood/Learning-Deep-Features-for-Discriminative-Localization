import tensorflow as tf
import numpy as np
import pandas as pd
import os

import tensorflow as tf

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


import os
import ipdb

weight_path = 'caffe_layers_value.pickle'
model_path = 'models/caltech256/'
n_epochs = 5
init_learning_rate = 0.01
weight_decay_rate = 0.0005
momentum = 0.9
batch_size = 60

dataset_path = '256_ObjectCategories'

caltech_path = 'data/caltech'
trainset_path = 'data/caltech/train.pickle'
testset_path = 'data/caltech/test.pickle'
label_dict_path = 'data/caltech/label_dict.pickle'

if not os.path.exists( trainset_path ):
    if not os.path.exists( caltech_path ):
        os.makedirs( caltech_path )
    image_dir_list = os.listdir( dataset_path )

    label_pairs = map(lambda x: x.split('.'), image_dir_list)
    labels, label_names = zip(*label_pairs)
    labels = map(lambda x: int(x), labels)

    label_dict = pd.Series( labels, index=label_names )
    label_dict -= 1
    n_labels = len( label_dict )

    image_paths_per_label = map(lambda one_dir: map(lambda one_file: os.path.join( dataset_path, one_dir, one_file ), os.listdir( os.path.join( dataset_path, one_dir))), image_dir_list)
    image_paths_train = np.hstack(map(lambda one_class: one_class[:-10], image_paths_per_label))
    image_paths_test = np.hstack(map(lambda one_class: one_class[-10:], image_paths_per_label))

    trainset = pd.DataFrame({'image_path': image_paths_train})
    testset  = pd.DataFrame({'image_path': image_paths_test })

    trainset = trainset[ trainset['image_path'].map( lambda x: x.endswith('.jpg'))]
    trainset['label'] = trainset['image_path'].map(lambda x: int(x.split('/')[-2].split('.')[0]) - 1)
    trainset['label_name'] = trainset['image_path'].map(lambda x: x.split('/')[-2].split('.')[1])

    testset = testset[ testset['image_path'].map( lambda x: x.endswith('.jpg'))]
    testset['label'] = testset['image_path'].map(lambda x: int(x.split('/')[-2].split('.')[0]) - 1)
    testset['label_name'] = testset['image_path'].map(lambda x: x.split('/')[-2].split('.')[1])

    label_dict.to_pickle(label_dict_path)
    trainset.to_pickle(trainset_path)
    testset.to_pickle(testset_path)
else:
    trainset = pd.read_pickle( trainset_path )
    testset  = pd.read_pickle( testset_path )
    label_dict = pd.read_pickle( label_dict_path )
    n_labels = len(label_dict)

learning_rate = tf.placeholder( tf.float32, [])
images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels')

modeon = modeltype(weight_path, n_labels)

p1,p2,p3,p4,conv5, conv6, gap, output = modeon.inference(images_tf)
loss_tf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output, labels_tf ))

weights_only = filter( lambda x: x.name.endswith('W:0'), tf.trainable_variables() )
weight_decay = tf.reduce_sum(tf.pack([tf.nn.l2_loss(x) for x in weights_only])) * weight_decay_rate
loss_tf += weight_decay

sess = tf.InteractiveSession()
saver = tf.train.Saver( max_to_keep=50)

optimizer = tf.train.MomentumOptimizer( learning_rate, momentum )
grads_and_vars = optimizer.compute_gradients( loss_tf )
grads_and_vars = map(lambda gv: (gv[0], gv[1]) if ('conv6' in gv[1].name or 'GAP' in gv[1].name) else (gv[0]*0.1, gv[1]), grads_and_vars)
#grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
train_op = optimizer.apply_gradients( grads_and_vars )
tf.initialize_all_variables().run()
# tf.global_variables_initializer().run()
sess.run(tf.initialize_all_variables())

testset.index  = range( len(testset) )

f_log = open('../results/results.txt', 'w')

iterations = 0
loss_list = []
for epoch in range(n_epochs):

    trainset.index = range( len(trainset) )
    trainset = trainset.ix[ np.random.permutation( len(trainset) )]

    for start, end in zip(
        range( 0, len(trainset)+batch_size, batch_size),
        range(batch_size, len(trainset)+batch_size, batch_size)):

        current_data = trainset[start:end]
        current_image_paths = current_data['image_path'].values
        current_images = np.array(map(lambda x: load_image(x), current_image_paths))

        good_index = np.array(map(lambda x: x is not None, current_images))

        current_data = current_data[good_index]
        current_images = np.stack(current_images[good_index])
        current_labels = current_data['label'].values

        _, loss_val, output_val = sess.run(
                [train_op, loss_tf, output],
                feed_dict={
                    learning_rate: init_learning_rate,
                    images_tf: current_images,
                    labels_tf: current_labels
                    })

        loss_list.append( loss_val )

        iterations += 1
        if iterations % 5 == 0:
            print "Epoch", epoch, "Iteration", iterations
            print "Processed", start, '/', len(trainset)

            label_predictions = output_val.argmax(axis=1)
            acc = (label_predictions == current_labels).sum()

            print "Accuracy:", acc, '/', len(current_labels)
            print "Training Loss:", np.mean(loss_list)
            loss_list = []

    n_correct = 0
    n_data = 0
    for start, end in zip(
            range(0, len(testset)+batch_size, batch_size),
            range(batch_size, len(testset)+batch_size, batch_size)
            ):
        current_data = testset[start:end]
        current_image_paths = current_data['image_path'].values
        current_images = np.array(map(lambda x: load_image(x), current_image_paths))

        good_index = np.array(map(lambda x: x is not None, current_images))

        current_data = current_data[good_index]
        current_images = np.stack(current_images[good_index])
        current_labels = current_data['label'].values

        output_vals = sess.run(
                output,
                feed_dict={images_tf:current_images})

        label_predictions = output_vals.argmax(axis=1)
        acc = (label_predictions == current_labels).sum()

        n_correct += acc
        n_data += len(current_data)

    acc_all = n_correct / float(n_data)
    f_log.write('epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n')
    print 'epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n'

    saver.save( sess, os.path.join( model_path, 'model'), global_step=epoch)

    init_learning_rate *= 0.99




