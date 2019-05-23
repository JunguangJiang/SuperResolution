import torch

import utility
import data
import model
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

print(args)
if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = None # TODO set a proper loss
    t = Trainer(args, loader, model, loss, checkpoint)
    t.test()

