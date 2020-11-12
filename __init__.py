from datasets.dataset_dota import DOTA
from models import ctrbox_net
import torch

if __name__ == '__main__':
    from torchsummary import summary
    import time
    import numpy as np
    dataset = {'dota': DOTA}
    num_classes = {'dota': 1, 'hrsc': 1}
    heads = {'hm': num_classes['dota'],
             'wh': 10,
             'reg': 2,
             'cls_theta': 1
             }
    down_ratio = 4
    model = ctrbox_net.CTRBOX7(heads=heads,
                               pretrained=True,
                               down_ratio=down_ratio,
                               final_kernel=1,
                               head_conv=32).cuda()
    time_s = []
    for i in range(10):
        input = torch.randn((2, 3, 512, 512)).cuda()
        # print("Total param size = %f MB" % (sum(v.numel() for v in model.parameters()) / 1024 / 1024))
        start_time = time.time()
        pr_decs = model(input)
        time_s.append(time.time() - start_time)
        print(time.time() - start_time)
    print(f'time consuming {np.mean(time_s)}')
    # print(pr_decs['hm'].size(), pr_decs['hm_aux'].size(), pr_decs['wh'].size(), pr_decs['reg'].size())