import torch


class NNConfig:
    """
    initialize various NN components from configuration
    """

    @staticmethod
    def get_optimizer(cfg, parameters):
        """
        static function
        initialize optimizer from configuration
        supported optimizers: Adam, AdamW, SGD
        raise KeyError is optimizer is not supported 
        or if required parameters are not provided
        """
        if not "optimizer" in cfg:
            raise KeyError("optimizer is not specified")

        # TODO: it will be nice to use match here
        if "Adam" in cfg.optimizer:
            if not "lr" in cfg.optimizer.Adam:
                raise KeyError("learning_rate not specified")
            return torch.optim.Adam(parameters, **cfg.optimizer.Adam)
        if "AdamW" in cfg.optimizer:
            if not "lr" in cfg.optimizer.AdamW:
                raise KeyError("learning_rate not specified")
            return torch.optim.AdamW(parameters, **cfg.optimizer.AdamW)
        if "SGD" in cfg.optimizer:
            if not "lr" in cfg.optimizer.SGD:
                raise KeyError("learning_rate not specified")
            return torch.optim.SGD(parameters, **cfg.optimizer.SGD)
        raise KeyError("provided optimizer is not supported")

    @staticmethod
    def get_scheduler(cfg, optimizer):
        """
        static function
        initialize scheduler from configuration
        supported schedulers: ReduceLROnPlateau
        raise KeyError is scheduler is not supported 
        or if required parameters are not provided
        """
        if not "scheduler" in cfg:
            raise KeyError("scheduler is not specified")
        if "ReduceLROnPlateau" in cfg.scheduler:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **cfg.scheduler.ReduceLROnPlateau
            )
        raise KeyError("provided scheduler is not supported")
