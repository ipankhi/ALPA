import timm
import torch
import torch.nn as nn

def get_model(args):
    model = timm.create_model('convnext_base', pretrained=True, num_classes=args.num_classes)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    if torch.cuda.is_available():
        device = torch.device(args.gpu_ids)
    else:
        device = torch.device("cpu")
    model = model.to(device)
    return model 

