from SRFBN_model import SRFBN
import tensorflow as tf
import time
from PreProcess import *

def train_SRFBN(datas, sess, cfg):
    # start put data in queue
    with tf.device("/cpu:0"):
        step = tf.Variable(0, trainable=False)
    srfbn = SRFBN(sess=sess, cfg=cfg)
    srfbn.train_step()

    ## build Optimizer
    boundaries = [len(datas)*epoch//cfg.batchsize for epoch in cfg.lr_steps]
    values = [cfg.learning_rate*(cfg.lr_gama**i) for i in range(len(cfg.lr_steps)+1)]
    lr = tf.train.piecewise_constant(step, boundaries, values)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    with tf.control_dependencies(update_ops):
        gs_vs = optimizer.compute_gradients(srfbn.losses)
        with tf.device("/cpu:0"):
            train_op = optimizer.apply_gradients(grads_and_vars=gs_vs, global_step=step)

    tf.global_variables_initializer().run(session=sess)

    summary_writer = tf.summary.FileWriter(cfg.srfbn_logdir, srfbn.sess.graph)
    if srfbn.cfg.load_premodel:
        counter = srfbn.load()
        ep = counter // (len(datas) // cfg.batchsize)
        it = counter % (len(datas) // cfg.batchsize)
    else:
        ep, it, counter = 0, 0, 0
    time_ = time.time()
    print("\nNow Start Training...\n")
    while ep < cfg.epoch:
        # Run by batch images
        iterate = len(datas) // cfg.batchsize
        while it < iterate:
            imgnames = random.choice(np.arange(len(datas)), 5)

            batch_labels, batch_images = preprocess(imgnames, cfg)

            _, err = srfbn.sess.run([train_op, srfbn.losses],
                                  feed_dict={srfbn.imageplaceholder: batch_images,
                                             srfbn.labelplaceholder: batch_labels})

            if it % 10 == 0:
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % (
                    (ep + 1), it, time.time() - time_, err))
            if it % 100 == 0:
                srfbn.save(counter)
                summary_str = srfbn.sess.run(srfbn.merged_summary,
                                             feed_dict={srfbn.imageplaceholder: batch_images,
                                             srfbn.labelplaceholder: batch_labels})
                summary_writer.add_summary(summary_str, counter)
            it += 1
            counter += 1
        it = 0
        ep += 1

def train(*args, **kwargs):
    txtfiles = kwargs["imgtxts"]
    imgs = []
    for txtfile in txtfiles:
        f = open(txtfile)
        tmp = f.readlines()
        imgs.extend(tmp)
        f.close()

    sess = tf.compat.v1.Session()

    ## build NetWork
    from config import SRFBN_config
    cfg = SRFBN_config()
    datas = imgs
    train_SRFBN(datas, sess, cfg)



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    imgtxts = [r"E:\Open_Datasets/SR_Images.txt"]

    train(imgtxts=imgtxts)