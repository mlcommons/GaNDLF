import torch
from torchmetrics import FID, SSIM, LPIPS
from sklearn.metrics import normalized_mutual_info_score 
import numpy as np

def generic_torchmetrics_score(output, label, metric_class, metric_key, params):
    #num_classes = params["model"]["num_classes"]
    #predicted_classes = output
    print(label.size())
    if metric_key == "fid":
        if label.size()[1] == 1:
            #special case for cycleGAN
            if params["model"]["architecture"] == "cycleGAN":
                label = (label + 1) / 2
                output = (output + 1) / 2            
            label = torch.cat((label,label,label),1)
            output = torch.cat((output,output,output),1)
            label = label*255
            output = output*255
            label = label.to(torch.uint8)
            output = output.to(torch.uint8)
            label = torch.cat((label,label),0)
            output = torch.cat((output,output),0)
        metric_function = metric_class(feature=64)
        metric_function.update(label.cpu(), real=True)
        metric_function.update(output.cpu(), real=False)
        return metric_function.compute()
    
    elif metric_key == "ssim":
        metric_function = metric_class()
        return metric_function(output.cpu(), label.cpu())
    
    elif metric_key == "lpips":
        if label.size()[1] == 1:
            label = torch.cat((label,label,label),1)
            output = torch.cat((output,output,output),1)
        metric_function = metric_class(net_type="vgg")
        return metric_function(output.cpu(), label.cpu())

    else:
        metric_function = metric_class()
        return metric_function(output.cpu(), label.cpu())


def FID_score(output, label, params):
    return generic_torchmetrics_score(output, label, FID, "fid", params)
    

def SSIM_score(output, label, params):
    return generic_torchmetrics_score(output, label, SSIM, "ssim", params)


def LPIPS_score(output, label, params):
    return generic_torchmetrics_score(output, label, LPIPS, "lpips", params)

def NMI_score(output, label, params):
    output = output[0].cpu().detach().numpy()
    label = label[0].cpu().detach().numpy()
    if label.shape[1] == 1:
        #special case for cycleGAN
        if params["model"]["architecture"] == "cycleGAN":
            label = (label + 1) / 2
            output = (output + 1) / 2
        label = label * 255
        output = output * 255
    label = label.astype(int)
    output = output.astype(int)
    output = output.ravel()
    label = label.ravel()
    return torch.tensor(normalized_mutual_info_score(output, label))
