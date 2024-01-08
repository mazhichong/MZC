import numpy as np
import tensorflow as tf
from utils import *
from activation import taylor_softmax
from RT_IDRNet import *
from tensorflow.python import pywrap_tensorflow

def get_RTIDR(len,DIR):

    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(allow_growth=True)
    )

    rtidr = {'dist':[], 'omega':[], 'theta':[], 'phi':[]}

    # network
    with tf.Graph().as_default():

        with tf.name_scope('input'):
            ncol = tf.placeholder(dtype=tf.int32, shape=())
        #IDRNet
        x = tf.random.normal([5, ncol, ncol,64])
        reg = tf.contrib.layers.l2_regularizer(1.5e-4)
        idrnet = IDRNet(reg_rate=reg).forward
        output_tensor = idrnet(x, layer=38)
        output_tensor = tf.nn.elu(tf.contrib.layers.instance_norm(output_tensor))
        #Predict residue distance and direction using Taylor_Softmax
        n = Taylor_Softmax(fliters=1, kernel_size=1, strides=(1, 1), padding='SAME',reg_rate=reg).forward
        prob_theta, prob_phi, prob_dist,prob_omega = n(output_tensor)

        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            for filename in os.listdir(DIR):
                if not filename.endswith(".index"):
                    continue
                mname = DIR+"/"+os.path.splitext(filename)[0]
                print('reading weights from:', mname)
                saver.restore(sess, mname)
                bd, bo, bt, bp = sess.run([prob_dist, prob_omega, prob_theta, prob_phi],feed_dict = { ncol : len })

                rtidr['dist'].append(np.mean(bd,axis=0))
                rtidr['theta'].append(np.mean(bt,axis=0))
                rtidr['omega'].append(np.mean(bo,axis=0))
                rtidr['phi'].append(np.mean(bp,axis=0))


    for key in rtidr.keys():
        rtidr[key] = np.mean(rtidr[key], axis=0)

    return rtidr
