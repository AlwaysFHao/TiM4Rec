# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/6/13
import sys

from recbole.config import Config
from logging import getLogger
from tim4rec import TiM4Rec
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.trainer import Trainer

from recbole.utils import (
    init_logger,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

if __name__ == '__main__':
    config = Config(model=TiM4Rec, config_file_list=[f'config/config4beauty_64d.yaml'])
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])

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
    model = TiM4Rec(config, train_data.dataset)
    model = model.to(config['device'])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = Trainer(config, model)
    if config['checkpoint_path'] is not None:
        trainer.resume_checkpoint(config['checkpoint_path'])

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
