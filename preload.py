import os
import torch
from torch import nn
from models.hrnet import hrnet


def preloader(seg_weights):
    current_path = os.path.dirname(os.path.abspath(__file__))

    model = hrnet(11)
    # if torch.cuda.is_available():
    #     model.cuda()
    model = nn.DataParallel(model)
    # print(torch.load(seg_weights,map_location=torch.device('cpu')).keys())
    model.load_state_dict(torch.load(
        seg_weights, map_location=torch.device('cpu'))["state_dict"])
    model.eval()

    return model
