import os

class config:
    def __init__(self):
        self.batchsize = 1
        self.Process_num = 3
        self.maxsize = 200
        self.ngpu = 1
        self.imagesize = 64
        self.scale = 3
        self.epoch = 1000
        self.checkpoint_dir = "./model"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.log_dir = "./log"
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.result = "./result"
        if not os.path.exists(self.result):
            os.mkdir(self.result)



class SRFBN_config(config):
    def __init__(self):
        super(SRFBN_config, self).__init__()
        self.istrain = True
        self.istest = not self.istrain
        self.c_dim = 3
        self.in_channels = 3
        self.out_channels = 3
        self.num_features = 32
        self.num_steps = 4
        self.num_groups = 6
        self.BN = True
        if self.BN:
            self.BN_type = "BN" # "BN" # or "IN"
        self.act_type = "prelu"
        self.loss_type = "L2"
        self.lr_steps = [150, 300, 550, 750]
        self.lr_gama = 1
        self.learning_rate = 2e-7
        self.load_premodel = True
        self.srfbn_logdir = "%s/srfbn" % self.log_dir
        if not os.path.exists(self.srfbn_logdir):
            os.mkdir(self.srfbn_logdir)
        self.srfbn_result = "%s/srfbn" % self.result
        if not os.path.exists(self.srfbn_result):
            os.mkdir(self.srfbn_result)
