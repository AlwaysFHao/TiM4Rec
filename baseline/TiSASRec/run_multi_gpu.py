import os
import sys
from logging import getLogger

import torch.multiprocessing as mp
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
from recbole.trainer import Trainer
import torch.distributed as dist
from collections.abc import MutableMapping

from TiSASRec import TiSASRec


def run_recbole(
        model=None,
        dataset=None,
        config_file_list=None,
        config_dict=None,
        saved=True,
        queue=None,
):
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = TiSASRec(config, train_data.dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = Trainer(config, model)
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

    if not config["single_spec"]:
        dist.destroy_process_group()

    if config["local_rank"] == 0 and queue is not None:
        queue.put(result)  # for multiprocessing, e.g., mp.spawn

    return result  # for the single process


def run_recboles(rank, *args):
    kwargs = args[-1]
    if not isinstance(kwargs, MutableMapping):
        raise ValueError(
            f"The last argument of run_recboles should be a dict, but got {type(kwargs)}"
        )
    kwargs["config_dict"] = kwargs.get("config_dict", {})
    kwargs["config_dict"]["local_rank"] = rank
    run_recbole(
        *args[:3],
        **kwargs,
    )


if __name__ == '__main__':
    os.chdir('../../')
    # Optional, only needed if you want to get the result of each process.
    queue = mp.get_context('spawn').SimpleQueue()

    config_dict = {}
    config_dict.update({
        "world_size": 8,
        "ip": 'local_host',
        "port": '5678',
        "nproc": 8,
        "offset": 0,
    })
    kwargs = {
        "config_dict": config_dict,
        "queue": queue,  # Optional
    }

    mp.spawn(
        run_recboles,
        args=(TiSASRec, 'ml-100k', ['config.yaml'], kwargs),
        nprocs=8,
        join=True,
    )

    # Normally, there should be only one item in the queue
    res = None if queue.empty() else queue.get()
