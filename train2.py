import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from data_utils import DatasetFromFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from tqdm import tqdm
from psnrmeter import PSNRMeter
import setting


class SREngine:
    def __init__(self):
        self.num_epochs = setting.num_epochs
        self.input_transform = transforms.Compose([
            transforms.Resize((270, 480)),
            transforms.ToTensor()
        ])
        self.target_transform = transforms.Compose([
            transforms.Resize((1080, 1920)),
            transforms.ToTensor()
        ])
        # TODO we need to add proper data augmentation
        self.train_set = DatasetFromFolder('data/image/train', input_transform=self.input_transform,
                                      target_transform=self.target_transform)
        self.val_set = DatasetFromFolder("data/image/valid", input_transform=self.input_transform,
                                    target_transform=self.target_transform)
        self.train_loader = DataLoader(dataset=self.train_set, num_workers=4, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(dataset=self.val_set, num_workers=4, batch_size=64,
                                shuffle=False)  # TODO shuffle False right?

        self.model = setting.model
        # TODO we could try different criterion
        self.criterion = nn.MSELoss()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=30, factor=0.85, verbose=True)

        self.engine = Engine()
        self.meter_loss = tnt.meter.AverageValueMeter()
        self.meter_psnr = PSNRMeter()
        self.best_meter_psnr = 0.0

        self.train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
        self.train_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Train PSNR'})
        self.val_loss_logger = VisdomPlotLogger('line', opts={'title': 'Val Loss'})
        self.val_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Val PSNR'})

        self.engine.hooks['on_sample'] = self.on_sample
        self.engine.hooks['on_forward'] = self.on_forward
        self.engine.hooks['on_start_epoch'] = self.on_start_epoch
        self.engine.hooks['on_end_epoch'] = self.on_end_epoch

    def run(self):
        self.engine.train(self.processor, self.train_loader, maxepoch=self.num_epochs, optimizer=self.optimizer)

    def processor(self, sample):
        data, target, training = sample
        data = Variable(data)
        target = Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = self.model(data)
        loss = self.criterion(output, target)

        return loss, output

    def on_sample(self, state):
        state['sample'].append(state['train'])

    def reset_meters(self):
        self.meter_psnr.reset()
        self.meter_loss.reset()

    def on_forward(self, state):
        self.meter_psnr.add(state['output'].data, state['sample'][1])
        self.meter_loss.add(state['loss'].data)

    def on_start_epoch(self, state):
        self.reset_meters()
        # self.scheduler.step()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(self, state):
        print('[Epoch %d] Train Loss: %.4f (PSNR: %.2f db)' % (
            state['epoch'], self.meter_loss.value()[0], self.meter_psnr.value()))

        self.train_loss_logger.log(state['epoch'], self.meter_loss.value()[0])
        self.train_psnr_logger.log(state['epoch'], self.meter_psnr.value())

        self.reset_meters()

        engine.test(self.processor, self.val_loader)
        self.val_loss_logger.log(state['epoch'], self.meter_loss.value()[0])
        self.val_psnr_logger.log(state['epoch'], self.meter_psnr.value())

        print('[Epoch %d] Val Loss: %.4f (PSNR: %.2f db)' % (
            state['epoch'], self.meter_loss.value()[0], self.meter_psnr.value()))
        # TODO Currently we only calculate PSNR without taking VMAF into consideration
        if self.best_meter_psnr < self.meter_loss.value()[0]:
            self.best_meter_psnr = self.meter_loss.value()[0]
            torch.save(self.model.state_dict(), 'model/{}.pt'.format(setting.model_name))


if __name__ == '__main__':
    engine = SREngine()
    engine.run()
