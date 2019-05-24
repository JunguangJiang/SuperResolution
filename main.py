import torch

import utility
import data
import model
from option import args
from trainer import Trainer
import loss

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

print(args)
if checkpoint.ok:
    loader = data.Data(args)
    _model = model.Model(args, checkpoint)
    if args.test_only:
        _loss = None
    else:
        _loss = loss.Loss(args, checkpoint)
    t = Trainer(args, loader, _model, _loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()
    checkpoint.done()
