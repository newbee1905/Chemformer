import inspect
from typing import List, Optional

import hydra
import math
import pytorch_lightning as pl
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import Plugin

from molbart.utils.callbacks.callback_collection import CallbackCollection
from molbart.utils.scores import ScoreCollection


def instantiate_callbacks(callbacks_config: Optional[DictConfig]) -> CallbackCollection:
    """Instantiates callbacks from config."""
    callbacks = CallbackCollection()
    if not callbacks_config:
        print("No callbacks configs found! Skipping...")
        return callbacks

    callbacks.load_from_config(callbacks_config)
    return callbacks


def instantiate_scorers(scorer_config: Optional[DictConfig]) -> ScoreCollection:
    """Instantiates scorer from config."""

    scorers = ScoreCollection()
    if not scorer_config:
        print("No scorer configs found! Skipping...")
        return scorers

    if not isinstance(scorer_config, DictConfig):
        raise TypeError("Scorer config must be a DictConfig!")

    for key, sc_conf in scorer_config.items():
        if isinstance(sc_conf, DictConfig) and "_target_" in sc_conf:
            print(f"Instantiating scorer <{sc_conf._target_}>")
            scorer_obj = hydra.utils.instantiate(sc_conf)
            scorers.add(key, scorer_obj)

    return scorers


def instantiate_logger(logger_config: Optional[DictConfig]) -> TensorBoardLogger:
    """Instantiates logger from config."""
    logger: TensorBoardLogger = []

    if not logger_config:
        print("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_config, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    if isinstance(logger_config, DictConfig) and "_target_" in logger_config:
        print(f"Instantiating logger <{logger_config._target_}>")
        logger = hydra.utils.instantiate(logger_config)

    return logger


def instantiate_plugins(plugin_cfg: Optional[DictConfig]) -> List[Plugin]:
    """Instantiates plugins from config."""
    plugins: list[Plugin] = []

    if not plugin_cfg:
        print("No plugin configs found! Skipping...")
        return plugins

    if not isinstance(plugin_cfg, (DictConfig, ListConfig)):
        raise TypeError("Plugin config must be a DictConfig or ListConfig!")

    if isinstance(plugin_cfg, DictConfig):
        config_list = plugin_cfg.values()
    else:
        config_list = plugin_cfg

    for plugin_conf in config_list:
        if isinstance(plugin_conf, DictConfig) and "_target_" in plugin_conf:
            print(f"Instantiating plugin <{plugin_conf._target_}>")
            plugins.append(hydra.utils.instantiate(plugin_conf))

    return plugins


def calc_train_steps(args, dm, n_gpus=1):
    n_gpus = getattr(args, "n_gpus", n_gpus)
    dm.setup()
    if n_gpus > 0:
        batches_per_gpu = math.ceil(len(dm.train_dataloader()) / float(n_gpus))
    else:
        raise ValueError("Number of GPUs should be > 0 in training.")
    train_steps = math.ceil(batches_per_gpu / args.acc_batches) * args.n_epochs
    return train_steps


def build_trainer(config, n_gpus=1):
    n_gpus = config.get("n_gpus", n_gpus)

    print("Instantiating loggers...")
    logger = instantiate_logger(config.get("logger"))

    print("Instantiating callbacks...")
    callbacks: CallbackCollection = instantiate_callbacks(config.get("callbacks"))

    print("Instantiating plugins...")
    plugins: list[Plugin] = instantiate_plugins(config.get("plugin"))

    print("Building trainer...")

    # Get all possible trainer arguments from pl.Trainer signature
    sig = inspect.signature(pl.Trainer.__init__)
    trainer_params = set(sig.parameters.keys())

    # Filter config to only include trainer arguments
    trainer_kwargs = {k: v for k, v in config.items() if k in trainer_params}

    # Renaming configuration to Trainer arguments
    if config.get("n_epochs") is not None:
        trainer_kwargs["max_epochs"] = config.get("n_epochs")
    if config.get("acc_batches") is not None:
        trainer_kwargs["accumulate_grad_batches"] = config.get("acc_batches")
    if config.get("clip_grad") is not None:
        trainer_kwargs["gradient_clip_val"] = config.get("clip_grad")

    trainer_kwargs["logger"] = logger
    trainer_kwargs["callbacks"] = callbacks.objects()
    trainer_kwargs["gpus"] = n_gpus
    if n_gpus > 1:
        trainer_kwargs["accelerator"] = "ddp"
        trainer_kwargs["plugins"] = plugins
    else:
        trainer_kwargs["plugins"] = None

    trainer: pl.Trainer = pl.Trainer(**trainer_kwargs)
    print("Finished trainer.")

    print(f"Default logging and checkpointing directory: {trainer.default_root_dir} or {trainer.weights_save_path}")
    return trainer

# vim: ts=4 sw=4 expandtab
