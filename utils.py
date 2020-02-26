import time

import torch
import torch.distributed as dist
import torch.utils.data.distributed
from apex import amp

from model import DeepSpeech


def reduce_tensor(tensor, world_size, reduce_op_max=False):
    rt = tensor.clone()
    dist.all_reduce(
        rt, op=dist.reduce_op.MAX if reduce_op_max is True else dist.reduce_op.SUM
    )  # Default to sum
    if not reduce_op_max:
        rt /= world_size
    return rt


def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    loss_valid = True
    error = ""
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = "WARNING: received a nan loss, setting loss value to 0"
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error


def load_model(device, model_path, use_half)->DeepSpeech:
    model:DeepSpeech = DeepSpeech.load_model(model_path)
    model.eval()
    model = model.to(device)
    if use_half:
        model = model.half()
    return model


def calc_loss(
    out,
    output_sizes,
    criterion,
    targets,
    target_sizes,
    device,
    is_distributed,
    world_size,
):
    float_out = out.transpose(0, 1).float()  # ensure float32 for loss
    loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
    loss = loss / out.size(0)  # average the loss by minibatch
    if is_distributed:
        loss = loss.to(device)
        loss_value = reduce_tensor(loss, world_size).item()
    else:
        loss_value = loss.item()

    return loss, loss_value


def train_one_epoch(
    model,
    train_loader,
    start_iter,
    train_sampler,
    data_time,
    batch_time,
    criterion,
    args,
    optimizer,
    epoch,
    device,
):
    end = time.time()
    avg_loss = 0
    for i, (data) in enumerate(train_loader, start=start_iter):
        if i == len(train_sampler):
            break
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = inputs.to(device)

        out, output_sizes = model(inputs, input_sizes)
        assert out.size(0) == inputs.size(0)
        loss, loss_value = calc_loss(
            out,
            output_sizes,
            criterion,
            targets,
            target_sizes,
            device,
            args.distributed,
            args.world_size,
        )

        # Check to ensure valid loss was calculated
        valid_loss, error = check_loss(loss, loss_value)
        if valid_loss:
            optimizer.zero_grad()
            # compute gradient

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
            optimizer.step()
        else:
            print(error)
            print("Skipping grad update")
            loss_value = 0

        avg_loss += loss_value

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if not args.silent and i % 100 == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss:.4f} ({loss:.4f})\t".format(
                    (epoch + 1),
                    (i + 1),
                    len(train_sampler),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_value,
                )
            )

    avg_loss /= i
    return avg_loss
