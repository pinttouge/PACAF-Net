import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from dataloder.dataset1718 import NPY_datasets
from utils import LogWritter
from loss_fn import ConfidentLoss
from tqdm import tqdm


class Solver():
    def __init__(self, module, opt, exp_id, train):
        self.opt = opt
        self.logger = LogWritter(opt)

        self.dev = torch.device("cuda:{}".format(opt.GPU_ID) if torch.cuda.is_available() else "cpu")
        self.net = module.Net(opt)
        if not train:
            self.net.load_state_dict(torch.load('D:/pytorchProj/PSNet/ckpt/best_for_isic18.pt'))
        self.net = self.net.to(self.dev)

        msg = "# params:{}\n".format(sum(map(lambda x: x.numel(), self.net.parameters())))
        print(msg)
        self.logger.update_txt(msg)

        self.loss_fn = ConfidentLoss(lmbd=opt.lmbda)

        # gather parameters
        base, head = [], []
        for name, param in self.net.named_parameters():
            if "encoder" in name:
                base.append(param)
            else:
                head.append(param)
        assert base != [], 'encoder is empty'
        self.optim = torch.optim.Adam([{'params': base}, {'params': head}], opt.lr, betas=(0.9, 0.999), eps=1e-8)

        self.exp_id = f'{opt.dataset}_{exp_id}'

        trainset = NPY_datasets(path_Data=opt.dataset_root, train=True)
        validset = NPY_datasets(path_Data=opt.dataset_root, train=False)

        self.train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True,
                                                        pin_memory=True, num_workers=0)

        self.eval_loader = torch.utils.data.DataLoader(dataset=validset, batch_size=1, shuffle=False, pin_memory=True,
                                                       num_workers=0)

        self.best_dice, self.best_step = 0, 0

    def fit(self):
        opt = self.opt

        for step in range(self.opt.max_epoch):
            epoch_start_time = time.time()
            #  assign different learning rate
            power = (step + 1) // opt.decay_step
            self.optim.param_groups[0]['lr'] = opt.lr * 0.1 * (0.5 ** power)  # for base
            self.optim.param_groups[1]['lr'] = opt.lr * (0.5 ** power)  # for head
            print('LR base: {}, LR head: {}'.format(self.optim.param_groups[0]['lr'],
                                                    self.optim.param_groups[1]['lr']))

            for i, inputs in enumerate(tqdm(self.train_loader)):
                self.optim.zero_grad()

                IMG = inputs[0].to(self.dev, dtype=torch.float)
                MASK = inputs[1].to(self.dev, dtype=torch.float32)

                pred = self.net(IMG)

                loss = self.loss_fn.get_value(pred, MASK)

                if i % 200 == 0:
                    print('iter: {}, loss: {}'.format(i, loss.item()))

                loss.backward()

                if opt.gclip > 0:
                    torch.nn.utils.clip_grad_value_(self.net.parameters(), opt.gclip)

                self.optim.step()
            # eval
            print("[{}/{}]".format(step + 1, self.opt.max_epoch))
            self.summary_and_save(step)
            epoch_end_time = time.time()
            print('epoch time: {}'.format(epoch_end_time - epoch_start_time))

    def summary_and_save(self, step):
        print('evaluate...')
        dice = self.evaluate()
        if dice > self.best_dice:
            self.best_dice = dice
            self.best_step = step
            # 保存模型
            self.save()

        print('epoch: {}, best_dice: {}, best_step: {}'.format(step, self.best_dice, self.best_step))

    @torch.no_grad()
    def evaluate(self):
        self.net.eval()

        gts = []
        preds = []

        for i, inputs in enumerate(tqdm(self.eval_loader)):
            IMG = inputs[0].to(self.dev, dtype=torch.float)
            MASK = inputs[1].to(self.dev, dtype=torch.float32)

            b, c, h, w = MASK.shape

            pred = self.net(IMG)

            pred_sal = F.pixel_shuffle(pred[-1], 4)  # from 64 to 256
            pred_sal = F.interpolate(pred_sal, (h, w), mode='bilinear', align_corners=False)
            pred_sal = torch.sigmoid(pred_sal)
            gts.append(MASK.squeeze(1).cpu().detach().numpy())
            preds.append(pred_sal.squeeze(1).cpu().detach().numpy())

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= 0.5, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        DSC = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        print('accuracy: {}, sensitivity: {}, specificity: {}, DSC: {}, miou: {}'.format(accuracy, sensitivity,
                                                                                               specificity, DSC,
                                                                                               miou))

        self.net.train()
        return DSC

    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state_dict)
        return

    def save(self):
        path = os.path.join(self.opt.ckpt_root, self.exp_id)
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, "best.pt")
        torch.save(self.net.state_dict(), save_path)
