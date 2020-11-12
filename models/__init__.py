from models.backbone import get_encoder
import torch

if __name__ == '__main__':
    encoder = get_encoder('densenet121')
    input = torch.randn((2, 3, 256, 256))
    print(encoder.out_shapes)
    outs = encoder(torch.randn((2, 3, 512, 512)))
    for out in outs:
        print(out.size())