import os


class LogWritter:
    def __init__(self, opt):
        self.root = opt.ckpt_root
        os.makedirs(self.root, exist_ok=True)

    def update_txt(self, msg, mode='a'):
        with open('{}/log.txt'.format(self.root), mode) as f:
            f.write(msg)