import torch
import torch.nn as nn
import threading
from torch._utils import ExceptionWrapper
import logging

def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None

def parallel_apply(fct, model, inputs, device_ids):
    modules = nn.parallel.replicate(model, device_ids)
    assert len(modules) == len(inputs)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input):
        torch.set_grad_enabled(grad_enabled)
        device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = fct(module, *input)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input))
                   for i, (module, input) in enumerate(zip(modules, inputs))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

def get_tcp_target(y_true, y_pred):
    """
    Calculate the TCP for each batch
    :param y_true: (batch_size, num_classes)
    :param y_pred: (batch_size, num_classes)
    :return: (batch_size)
    """
    tcp_target = []
    for i in range(y_true.shape[0]):    # for each batch
        tcp = 0.0
        for j in range(y_true[i].shape[0]): # for each class
            tcp += y_pred[i][j] * y_true[i][j]

        tcp = tcp / torch.count_nonzero(y_true[i]) if torch.count_nonzero(y_true[i]) != 0 else 0.0
        tcp_target.append(tcp)
    
    return torch.tensor(tcp_target).to(y_true.device)

def binary_ce(output1, output2):
    """
    Binary Cross Entropy Loss
    :param output1:
    :param output2:
    :return:
    """
    loss_bce = nn.BCEWithLogitsLoss(reduction='mean')
    loss = [loss_bce(output1[i], output2[i]) for i in range(len(output1))]
    return loss