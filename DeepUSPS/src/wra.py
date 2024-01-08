import tensorflow as tf
import numpy as np
import sys
import time
from activation import taylor_softmax
from utils import *
from SP_IDRNet import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda,Input,Conv2D,Layer

def wra(DIR,seq0,rtidr,aa_weight):

    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(allow_growth=True)
    )

    GRAD=[0,0]
    A=[np.pi/2,np.pi/2]
    L = len(seq0)
    traj = []

    # network网络

    with tf.Graph().as_default():

        # inputs
        with tf.name_scope('input'):
            ncol = L
            xy = tf.placeholder(dtype=tf.float32, shape=(None, None, None))

        # RTIDR distributions
        rd = tf.constant(rtidr['dist'], dtype=tf.float32)
        ro = tf.constant(rtidr['omega'], dtype=tf.float32)
        rt = tf.constant(rtidr['theta'], dtype=tf.float32)
        rp = tf.constant(rtidr['phi'], dtype=tf.float32)

        # aa bkgr composition in natives
        aa_bkgr = tf.constant(np.array([0.07892653, 0.04979037, 0.0451488 , 0.0603382 , 0.01261332,
                                        0.03783883, 0.06592534, 0.07122109, 0.02324815, 0.05647807,
                                        0.09311339, 0.05980368, 0.02072943, 0.04145316, 0.04631926,
                                        0.06123779, 0.0547427 , 0.01489194, 0.03705282, 0.0691271 ]),dtype=tf.float32)

        xy=tf.convert_to_tensor(xy, dtype=tf.float32)
        x1 = taylor_softmax(xy)
        x2 = tf.argmax(x1, -1)   #argmax
        x3 = tf.one_hot(x2, 20, dtype=tf.float32)    # convert inputs to 1-hot
        msa1hot1 = tf.stop_gradient(x3 - x1) + x1   #tf.stop_gradient

        def add_gap(x):
            return tf.pad(x, [[0, 0], [0, 0],[0, 1]])
        msa1hot=add_gap(msa1hot1)

        #extract 1D、2D freature
        f2d = PSSM(diag=0.4)([msa1hot1, msa1hot])

        # store ensemble of networks in separate branches
        preds = [[] for _ in range(4)]

        # IDRnet
        reg = tf.contrib.layers.l2_regularizer(1.5e-4)
        idrnet = IDRNet(reg_rate=reg).forward
        output_tensor = idrnet(f2d,50)
        output_tensor = tf.nn.elu(tf.contrib.layers.instance_norm(output_tensor))

        # Taylor-softmax ---> Predict residue distance and direction using Taylor_Softmax
        pred = Taylor_Softmax(fliters=1, kernel_size=1, strides=(1, 1), padding='SAME', reg_rate=reg).forward
        prob_theta, prob_phi, prob_dist, prob_omega = pred(output_tensor)


        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            for filename in os.listdir(DIR):
                if not filename.endswith(".index"):
                    continue
                mname = DIR + "/" + os.path.splitext(filename)[0]
                print('reading weights from:', mname)
                saver.restore(sess, mname)
                sts, sps, sds, sos = prob_theta, prob_phi, prob_dist, prob_omega

                preds[0].append(sts[0])
                preds[1].append(sps[0])
                preds[2].append(sds[0])
                preds[3].append(sos[0])
            st = tf.reduce_mean(tf.stack(preds[0]), axis=0)
            sp = tf.reduce_mean(tf.stack(preds[1]), axis=0)
            sd = tf.reduce_mean(tf.stack(preds[2]), axis=0)
            so = tf.reduce_mean(tf.stack(preds[3]), axis=0)

            loss_dist = -tf.math.reduce_mean(tf.math.reduce_sum(sd * tf.math.log(sd / rd), axis=-1))
            loss_omega = -tf.math.reduce_mean(tf.math.reduce_sum(so * tf.math.log(so / ro), axis=-1))
            loss_theta = -tf.math.reduce_mean(tf.math.reduce_sum(st * tf.math.log(st / rt), axis=-1))
            loss_phi = -tf.math.reduce_mean(tf.math.reduce_sum(sp * tf.math.log(sp / rp), axis=-1))

            # aa composition loss aa
            aa_samp = tf.reduce_sum(msa1hot[0, :, :20], axis=0) / tf.cast(ncol, dtype=tf.float32) + 1e-7
            aa_samp = aa_samp / tf.reduce_sum(aa_samp)
            loss_aa = tf.reduce_sum(aa_samp * tf.log(aa_samp / aa_bkgr))

            # total loss
            loss = loss_dist + loss_omega + loss_theta + loss_phi + aa_weight * loss_aa


            # grad
            grad = tf.gradients(loss, xy)[0]

            #initial pssm
            seq = AA2Num(seq0).copy().reshape([1, L])
            seq_start = 0.8 * np.eye(21)[seq][:, :, :20]
            aa1 = np.random.normal(0, 0.01, size=(1, L, 20))
            aa=aa1+seq_start

            best_loss, best_I, K = np.inf, None, 0
            mt, vt, b1, b2 = 0, 0, 0.9, 0.999,

            #optimate sequence
            for j in range(2100):
                AA = np.copy(aa)
                loss1, grad1 = sess.run([loss, grad], feed_dict={xy: AA})
                if loss1 < best_loss:
                    best_loss, best_I, K = loss1, np.copy(Num2AA(AA[0].argmax(-1))), j+1

                abc = Num2AA(AA[0].argmax(-1))
                print("%8d %s %.6f" % (j, abc, loss1))

                # AngularGrad
                del GRAD[0]
                GRAD.append(grad1)
                del A[0]
                AB=np.abs((GRAD[1]-GRAD[0])/(1+GRAD[1]*GRAD[0]))
                A1=np.arctan(AB)
                A.append(A1)
                ANS=np.less(A[0], A[1])
                mean_ans = np.mean(ANS.astype("float32"))
                if mean_ans > 0.5:
                    Amin = A[0]
                else:
                    Amin = A[1]
                A2=np.abs(np.tan(Amin))
                A3=np.tanh(A2)*1/2+1/2
                mt = b1 * mt + (1 - b1) * grad1
                vt = b2 * vt + (1 - b2) * np.square(grad1).sum((-1, -2), keepdims=True)
                mt1 = mt/(1-np.power(b1,j+1))
                vt1 = vt/(1-np.power(b2,j+1))
                grad1 =  A3*mt1/(np.sqrt(vt1) + 1e-8)

                #learning rate(warm restart)
                if 0 <= j < 300:
                    jj = j % 300
                    lr = 1/2 * 1.5 * (1+np.cos(np.pi*jj/300))
                elif 300 <= j < 900:
                    jj = (j - 300) % 600
                    lr = 1/2 * 0.7 * (1 + np.cos(np.pi * jj / 600))
                else:
                    jj = (j - 900) % 1200
                    lr = 1/2 * 0.3 * (1 + np.cos(np.pi * jj / 1200))
                aa -= lr * grad1
                traj.append([j, abc, loss1])

            print("%8d %s %.6f" % (K, best_I, best_loss))

    return traj,abc
