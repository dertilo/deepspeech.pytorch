import torch
import torch.distributed as dist

from model import DeepSpeech


def reduce_tensor(tensor, world_size, reduce_op_max=False):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.MAX if reduce_op_max is True else dist.reduce_op.SUM)  # Default to sum
    if not reduce_op_max:
        rt /= world_size
    return rt


def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    loss_valid = True
    error = ''
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error


def load_model(device, model_path, use_half):
    model = DeepSpeech.load_model(model_path)
    model.eval()
    model = model.to(device)
    if use_half:
        model = model.half()
    return model


def forward_and_calc_loss(model,inputs,input_sizes,criterion,targets,target_sizes,device,is_distributed,world_size):
    out, output_sizes = model(inputs, input_sizes)
    float_out = out.transpose(0, 1).float()  # ensure float32 for loss
    loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
    loss = loss / inputs.size(0)  # average the loss by minibatch
    if is_distributed:
        loss = loss.to(device)
        loss_value = reduce_tensor(loss, world_size).item()
    else:
        loss_value = loss.item()

    return out,output_sizes, loss, loss_value
