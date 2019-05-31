import sys
import torch
from collections import OrderedDict
"""
python net_interpolation.py [model1_path] [model2_path] [net_interp_path] [alpha]
e.g. python net_interpolation.py ../../experiment/baseline_edsr_mse2/model/model_best.pt ../../experiment/esrgan/model/model_best.pt ../../experiment/interpolation/model.pt 0.2

"""


def net_interpolation(model1_path, model2_path, net_interp_path, alpha):
    """
    interpolate the parameters of the model1 and model2
    eg. p1 belongs to model1, p2 belongs to model2
    then (1-alpha)*p1 + alpha*p2 belongs to the interpolated model
    :param model1_path:
    :param model2_path:
    :param net_interp_path:
    :param alpha:
    :return:
    """
    net1 = torch.load(model1_path)
    net2 = torch.load(model2_path)
    net_interp = OrderedDict()

    for k, v1 in net1.items():
        v2 = net2[k]
        net_interp[k] = (1-alpha) * v1 + alpha * v2
    torch.save(net_interp, net_interp_path)


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(__doc__)
    else:
        net_interpolation(model1_path=sys.argv[1], model2_path=sys.argv[2],
                          net_interp_path=sys.argv[3], alpha=sys.argv[4])
