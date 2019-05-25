import os
import math
from preprocess.preprocess import y4m_2_bmp, bmp_2_y4m

import utility
from data import common
import model
from option import args
import data

import torch
from tqdm import tqdm


class VideoTester():
    def __init__(self, args, loader, my_model, ckp):
        self.args = args
        self.loader_test = loader.loader_test
        self.scale = args.scale[0]
        self.ckp = ckp
        self.model = my_model

    def test(self):
        torch.set_grad_enabled(False)
        self.ckp.write_log('\nEvaluation on test:')
        psnr_score = 0
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results:
            self.ckp.begin_background()

        for idx_data, (lr, filename) in enumerate(tqdm(self.loader_test)):
            lr, = self.prepare(lr)
            sr = self.model(lr, self.scale)
            sr = utility.quantize(sr, self.args.rgb_range)

            save_list = [sr]
            if self.args.save_results:
                self.ckp.save_results(self.loader_test, filename[0], save_list, self.scale)

        size = len(self.loader_test)
        if size > 0:
            psnr_score /= len(self.loader_test)
        else:
            self.ckp.write_log("no test data!!!")
            exit(-1)

        if self.args.save_results:
            self.ckp.end_background()

        self.ckp.write_log("PSNR:{:.3f}".format(psnr_score))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else self.args.cuda)
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    args.test_only = True
    model = model.Model(args, checkpoint)
    loader = data.Data(args)
    t = VideoTester(args, loader, model, checkpoint)
    t.test()

