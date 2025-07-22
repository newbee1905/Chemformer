""" Module containing classes used to score the reaction routes.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import hydra
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import Callback

from molbart.utils.base_collection import BaseCollection

if TYPE_CHECKING:
    from typing import List


class CallbackCollection(BaseCollection):
    """
    Store callback objects for the chemformer model.

    The callbacks can be obtained by name

    .. code-block::

        callbacks = CallbackCollection()
        callback = callbacks['LearningRateMonitor']
    """

    _collection_name = "callbacks"

    def __init__(self) -> None:
        super().__init__()
        self._logger = logging.Logger("callback-collection")

    def __repr__(self) -> str:
        if self.selection:
            return f"{self._collection_name} ({', '.join(self.selection)})"

        return f"{self._collection_name} ({', '.join(self.items)})"

    def load(self, callback: Callback) -> None:  # type: ignore
        """
        Add a pre-initialized callback object to the collection

        Args:
            callback: the item to add
        """
        if not isinstance(callback, Callback):
            raise ValueError(
                "Only objects of classes inherited from " "pytorch_lightning.callbacks.Callbacks can be added"
            )
        self._items[repr(callback)] = callback
        self._logger.info(f"Loaded callback: {repr(callback)}")

    def load_from_config(self, callbacks_config: [DictConfig | ListConfig]) -> None:
        """
        Load one or several callbacks from a configuration dictionary

        Args:
            callbacks_config: Config of callbacks. Can be a DictConfig or a ListConfig.
        """
        if not isinstance(callbacks_config, (DictConfig, ListConfig)):
            self._logger.warning("Callbacks config is not a DictConfig or ListConfig, skipping.")
            return

        if isinstance(callbacks_config, DictConfig):
            config_list = callbacks_config.values()
        else:
            config_list = callbacks_config

        for cb_conf in config_list:
            if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
                self._logger.info(f"Instantiating callback <{cb_conf._target_}>")
                obj = hydra.utils.instantiate(cb_conf)
                self.load(obj)
