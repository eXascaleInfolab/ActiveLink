from collections import OrderedDict
import copy
import datetime
import logging

import torch
from torch.autograd import Variable

from evaluation import ranking_and_hits

log = logging.getLogger()


def early_stopping(mr, mr_opt, threshold):
    return 100 * (mr / mr_opt - 1) > threshold


def perform_task_meta_update(config, str2var_val, model, updated_params):
    e1_val = str2var_val["e1"]
    rel_val = str2var_val["rel"]
    e2_multi_val = str2var_val["e2_multi1_binary"].float()
    e2_multi_val = ((1.0 - config.label_smoothing_epsilon) * e2_multi_val) + (1.0 / e2_multi_val.size(1))

    task_pred = model.forward(e1_val, rel_val, weights=updated_params)
    task_loss = model.loss(task_pred, e2_multi_val)
    task_grads = torch.autograd.grad(task_loss, updated_params.values())

    model_pred = model.forward(e1_val, rel_val)
    model_loss = model.loss(model_pred, e2_multi_val)
    model_grads = torch.autograd.grad(model_loss, model.parameters(), create_graph=True)

    meta_grads = {}
    for task_grad, model_grad, (name, param) in zip(task_grads, model_grads, model.named_parameters()):
        shape = model_grad.shape
        task_grad.volatile = False

        if name == 'emb_rel.weight':
            model_second_grad = Variable(torch.cuda.FloatTensor(*shape).fill_(0)), # emb_rel gradient is constant => 2nd gradient is 0
        else:
            if len(shape) > 1:
                new_shape = 1
                for dim in shape:
                    new_shape = new_shape * dim

                task_grad = task_grad.view(new_shape)
                model_grad = model_grad.view(new_shape)

            g = torch.dot(model_grad, task_grad)

            model_second_grad = torch.autograd.grad(g, param, retain_graph=True)

            if task_grad.shape != shape:
                task_grad = task_grad.view(*shape)

        final_grad = task_grad - torch.mul(model_second_grad[0], config.inner_learning_rate)
        meta_grads[name] = final_grad
    return meta_grads


def perform_meta_update(config, val_batcher, model, grads, opt):
    log.info("Perform meta update")

    try:
        str2var = val_batcher.next()
    except StopIteration:
        str2var = val_batcher.next()
    e1 = str2var["e1"]
    rel = str2var["rel"]
    e2_multi = str2var["e2_multi1_binary"].float()
    e2_multi = ((1.0 - config.label_smoothing_epsilon) * e2_multi) + (1.0 / e2_multi.size(1))

    # We use a dummy forward / backward pass to get the correct grads into self.net
    pred = model.forward(e1, rel)
    loss = model.loss(pred, e2_multi)

    # Unpack the list of grad dicts
    gradients = {k: sum(g[k] for g in grads) for k in grads[0].keys()}

    # Register a hook on each parameter in the net that replaces the current dummy grad
    # with our grads accumulated across the meta-batch
    hooks = []
    for (k, v) in model.named_parameters():
        def get_closure():
            key = k

            def replace_grad(grad):
                return gradients[key]

            return replace_grad

        hooks.append(v.register_hook(get_closure()))

    # Compute grads for current step, replace with summed gradients as defined by hook
    opt.zero_grad()
    loss.backward()

    # Update the net parameters with the accumulated gradient according to optimizer
    opt.step()

    pred = model.forward(e1, rel)
    loss = model.loss(pred, e2_multi)

    # Remove the hooks before next training phase
    for h in hooks:
        h.remove()

    return loss


def run_inner(config, model, task):
    try:
        str2var = task.next()
    except:
        str2var = task.next()
    e1 = str2var["e1"]
    rel = str2var["rel"]
    e2_multi = str2var["e2_multi1_binary"].float()
    e2_multi = ((1.0 - config.label_smoothing_epsilon) * e2_multi) + (1.0 / e2_multi.size(1))

    pred = model.forward(e1, rel)
    loss = model.loss(pred, e2_multi)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    updated_params = OrderedDict()
    for (name, param), grad in zip(model.named_parameters(), grads):
        param_prime = copy.deepcopy(param)
        param_prime.data.sub_(config.inner_learning_rate * grad.data)
        updated_params[name] = param_prime

    # Compute the meta gradient and return it
    try:
        str2var_val = task.next()
    except:
        str2var_val = task.next()
    meta_grads = perform_task_meta_update(config, str2var_val, model, updated_params)

    return meta_grads


def run_meta_incremental(config, model, train_batcher, test_rank_batcher):
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    mr_opt = float('inf')
    model_opt = model
    early_stop_flag = False
    i = 1

    while True:
        log.info("{} epoch: started".format(i))
        epoch_start_time = datetime.datetime.now()
        model.train()

        grads = []

        log.info("Inner loop: started")
        inner_loop_start = datetime.datetime.now()
        for task in train_batcher:
            g = run_inner(config, model, task)
            grads.append(g)
        log.info("Inner loop: finished")
        log.info("Inner loop took {}".format(datetime.datetime.now() - inner_loop_start))

        # Perform the meta update
        perform_meta_update(config, test_rank_batcher, model, grads, opt)

        if i % config.eval_rate == 0:
            model.eval()
            log.info("Evaluation: started")
            eval_start_time = datetime.datetime.now()
            mr = ranking_and_hits(model, test_rank_batcher, config.batch_size, 'test_evaluation')
            log.info("Evaluation: finished")
            log.info("Evaluation took {}".format(datetime.datetime.now() - eval_start_time))

            early_stop_flag = early_stopping(mr, mr_opt, config.early_stop_threshold)

            if mr < mr_opt:
                model_opt = model
                mr_opt = mr

        log.info("{} epoch: finished".format(i))
        log.info("Epoch {} took {}".format(i, datetime.datetime.now() - epoch_start_time))

        if early_stop_flag:
            log.info("Early stopping after {} epochs".format(i))
            break

        i += 1

    return model_opt
