import json
from addict import Dict


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
        else:
            return value
        raise ex


def get_default_config():
    input = """{
        "batch_size":64,
        "optimizer": {
            "Adam": {
                "lr":0.01
                }
            },
        "scheduler": {
                "ReduceLROnPlateau": {
                    "factor": 0.1, 
                    "mode":"max", 
                    "verbose":true
                },
                "extra": {
                    "monitor": "val_acc"
                }
        }
    }
    """
    cfg = ConfigDict(json.loads(input))
    return cfg
