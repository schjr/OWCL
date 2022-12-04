import random
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .utils import AverageMeter


def uncr_generator(args, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    pseudo_idx = []
    pseudo_maxstd = []
    model.eval()

    data_loader = tqdm(data_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets, indexs) in enumerate(data_loader): # 146*256
            inputs1 = inputs[0].cuda()
            inputs2 = inputs[1].cuda()
            inputs3 = inputs[2].cuda()
            inputs4 = inputs[3].cuda()
            inputs5 = inputs[4].cuda()
            inputs6 = inputs[5].cuda()
            inputs7 = inputs[6].cuda()
            inputs8 = inputs[7].cuda()
            inputs9 = inputs[8].cuda()
            inputs10 = inputs[9].cuda()
            targets = targets.cuda()
            out_prob = []

            outputs,_ = model(inputs1)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs,_ = model(inputs2)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs,_ = model(inputs3)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs,_ = model(inputs4)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs,_ = model(inputs5)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs,_ = model(inputs6)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs,_ = model(inputs7)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs,_ = model(inputs8)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs,_ = model(inputs9)
            out_prob.append(F.softmax(outputs, dim=1))

            outputs,_ = model(inputs10)
            out_prob.append(F.softmax(outputs, dim=1))

            # compute uncertainty scores
            out_prob = torch.stack(out_prob)
            out_std = torch.std(out_prob, dim=0)
            out_prob = torch.mean(out_prob, dim=0)
            _, max_idx = torch.max(out_prob, dim=1)
            max_std = out_std.gather(1, max_idx.view(-1,1))

            pseudo_maxstd.extend(max_std.squeeze(1).cpu().numpy().tolist())
            pseudo_idx.extend(indexs.numpy().tolist())


            batch_time.update(time.time() - end)
            end = time.time()
            data_loader.set_description("UncrGen Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s.".format(
                batch=batch_idx + 1,
                iter=len(data_loader),
                data=data_time.avg,
                bt=batch_time.avg,
            ))

        data_loader.close()

    pseudo_maxstd = np.array(pseudo_maxstd)
    pseudo_idx = np.array(pseudo_idx)

    # normalizing the uncertainty values
    pseudo_maxstd = pseudo_maxstd/max(pseudo_maxstd)
    pseudo_maxstd = np.clip(pseudo_maxstd, args.temperature, 1.0)
    
    uncr_temp = {'index': pseudo_idx.tolist(), 'uncr': pseudo_maxstd.tolist()} # len(37500)
    return uncr_temp