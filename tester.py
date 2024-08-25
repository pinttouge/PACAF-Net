import os
import skimage.io as io
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloder.dataset1718 import NPY_datasets
class Tester():
    def __init__(self, module, opt):
        self.opt = opt

        self.dev = torch.device("cuda:{}".format(opt.GPU_ID) if torch.cuda.is_available() else "cpu")
        self.net = module.Net(opt)
        self.net = self.net.to(self.dev)

        msg = "# params:{}\n".format(
            sum(map(lambda x: x.numel(), self.net.parameters())))
        print(msg)
        validset = NPY_datasets(path_Data=opt.dataset_root, train=False)

        self.test_loader = torch.utils.data.DataLoader(dataset=validset, batch_size=1, shuffle=False, pin_memory=True,
                                                       num_workers=0)

    @torch.no_grad()
    def evaluate(self, path):
        opt = self.opt
        try:
            print('loading model from: {}'.format(path))
            self.load(path)
        except Exception as e:
            print(e)

        self.net.eval()

        if opt.save_result:
            save_root = os.path.join(opt.save_root, opt.save_msg)
            os.makedirs(save_root, exist_ok=True)

        for i, inputs in enumerate(tqdm(self.test_loader)):
            IMG = inputs[0].to(self.dev, dtype=torch.float)
            MASK = inputs[1].to(self.dev, dtype=torch.float32)
            NAME = inputs[2][0]

            b, c, h, w = MASK.shape

            pred = self.net(IMG)

            mask = (MASK * 255.).squeeze().detach().cpu().numpy().astype('uint8')
            pred_sal = F.pixel_shuffle(pred[-1], 4)
            pred_sal = F.interpolate(pred_sal, (h, w), mode='bilinear', align_corners=False)
            pred_sal = torch.sigmoid(pred_sal).squeeze()
            threshold = 0.5
            pred_sal = (pred_sal > threshold).float()
            pred_sal = (pred_sal * 255.).detach().cpu().numpy().astype('uint8')

            if opt.save_result:
                save_path_msk = os.path.join(save_root, "{}_msk.png".format(NAME))
                io.imsave(save_path_msk, mask)

                if opt.save_all:
                    for idx, sal in enumerate(pred[1:]):
                        scale = 256 // (sal.shape[-1])
                        sal_img = F.pixel_shuffle(sal, scale)
                        sal_img = F.interpolate(sal_img, (h, w), mode='bilinear', align_corners=False)
                        sal_img = torch.sigmoid(sal_img)
                        sal_path = os.path.join(save_root, "{}_sal_{}.png".format('mask', idx))
                        sal_img = sal_img.squeeze().detach().cpu().numpy()
                        sal_img = (sal_img * 255).astype('uint8')
                        io.imsave(sal_path, sal_img)
                else:
                    # save pred image
                    save_path_sal = os.path.join(save_root, "{}_sal.png".format(NAME))
                    io.imsave(save_path_sal, pred_sal)
            # mae += calculate_mae(mask, pred_sal)

        return 0

    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state_dict)
        return
