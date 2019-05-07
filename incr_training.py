import datetime
import logging

import torch

from evaluation import ranking_and_hits

log = logging.getLogger()


def early_stopping(mr, mr_opt, threshold):
    return 100 * (mr / mr_opt - 1) > threshold


def run_incremental(model, config, train_batcher, test_rank_batcher):
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    mr_opt = float('inf')
    model_opt = model
    early_stop_flag = False
    i = 1

    while True:
        log.info("{} epoch: started".format(i))
        epoch_start_time = datetime.datetime.now()

        model.train()
        for j, str2var in enumerate(train_batcher):
            opt.zero_grad()
            e1 = str2var["e1"]
            rel = str2var["rel"]
            e2_multi = str2var["e2_multi1_binary"].float()
            # label smoothing
            e2_multi = ((1.0 - config.label_smoothing_epsilon) * e2_multi) + (1.0 / e2_multi.size(1))

            pred = model.forward(e1, rel)
            loss = model.loss(pred, e2_multi)
            loss.backward()
            opt.step()

        if i % config.eval_rate == 0:
            model.eval()
            mr = ranking_and_hits(model, test_rank_batcher, config.batch_size, 'dev_evaluation')

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
