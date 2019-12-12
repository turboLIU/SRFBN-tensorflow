import os, cv2
import tensorflow as tf
import numpy as np
from SRFBN_model import SRFBN


def test_SRFBN(image):
    from config import SRFBN_config
    cfg = SRFBN_config()
    height, width, _ = image.shape()

    gpucfg = tf.ConfigProto()
    gpucfg.gpu_options.allow_growth = True
    sess = tf.Session(config=gpucfg)

    srfbn = SRFBN(sess, cfg)
    out = srfbn.test(width, height)
    tf.global_variables_initializer().run(session=sess)
    srfbn.saver = tf.train.Saver(max_to_keep=1)
    srfbn.load()
    cv2.namedWindow("result", 0)


    img = image.reshape([1, height, width, 3])
    output = sess.run(out, feed_dict={srfbn.imageplaceholder: img})
    output = output[0] * 128 + 127.5

    cv2.imshow("result", np.uint8(output))
    cv2.waitKey(0)
    srfbn.sess.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    image = r'./data/testdatas/s1.JPG'
    img = cv2.imread(image)
    img = (img - 127.5) / 128

    test_SRFBN(img)
