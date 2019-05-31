import os
import math
from decimal import Decimal

import logger
import utility
import torch
import torch.nn.utils as utils
import torch.nn as nn
from tqdm import tqdm


class Trainer():
    def __init__(self, args, loader, model, ema_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale[0]  # 我们只支持一种放大尺度的训练

        self.ckp = ckp
        self.loader_train = loader.loader_train  # 训练数据加载器
        self.loader_test = loader.loader_test  # 测试数据加载器
        self.model = model
        self.ema_model = ema_model
        for param in ema_model.parameters():
            param.detach_()

        self.loss = my_loss
        self.consistency_criterion = nn.MSELoss()
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':  # 恢复之前的训练
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        self.iter = 0

        self.vis = logger.Logger('gg', 'vis/gg1')

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        batch_loss = 0
        for batch, (lr, hr, _) in enumerate(tqdm(self.loader_train)):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.iter += 1

            self.optimizer.zero_grad()
            sr = self.model(lr, self.scale)

            sr_ema = self.ema_model(lr, self.scale)
            ema_alpha = self.update_ema_params(self.args.ema_decay)

            consistency_weight = self.get_current_consistency_weight()
            consistency_loss = self.consistency_criterion(sr, sr_ema) * consistency_weight


            loss = self.loss(sr, hr)

            self.vis.add_scalar('loss', loss)
            self.vis.add_scalar('consist loss', consistency_loss)
            self.vis.add_scalar('consistency_weight', consistency_weight)
            self.vis.add_scalar('alpha', ema_alpha)
            self.vis.step(1)

            loss += consistency_loss
            batch_loss += loss
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
                print("alpha: ", ema_alpha)

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        # self.optimizer.schedule(batch_loss/len(self.loader_train))

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch() + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1)
        )
        self.ema_model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, (lr, hr, filename) in enumerate(tqdm(self.loader_test)):
            lr, hr = self.prepare(lr, hr)
            sr = self.ema_model(lr, self.scale)
            sr = utility.quantize(sr, self.args.rgb_range)

            save_list = [sr]
            self.ckp.log[-1] += utility.calc_psnr(
                sr, hr, self.scale, self.args.rgb_range, dataset=self.loader_test
            )
            if self.args.save_gt:
                save_list.extend([lr, hr])

            if self.args.save_results:
                self.ckp.save_results(self.loader_test, filename[0], save_list, self.scale)

        self.ckp.log[-1] /= len(self.loader_test)
        best = self.ckp.log.max(0)
        self.ckp.write_log(
            '[{} x{}]\tPSNR: {:.3f} @epoch{} (Best: {:.3f} @epoch {})'.format(
                self.loader_test.dataset.name,
                self.scale,
                self.ckp.log[-1],
                epoch,
                best[0],
                best[1] + 1
            )
        )
        self.optimizer.schedule(self.ckp.log[-1])

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))

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

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

    def update_ema_params(self, alpha):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.iter + 1), alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        return alpha

    def get_current_consistency_weight(self):
        return min(self.args.consistency * utility.sigmoid_rampup(self.iter, self.args.consistency_rampup), 1)