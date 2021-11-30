from torch.utils.data import DataLoader
import torch

import fastspeech.datasets
from fastspeech.collate_fn.collate import collate_fn
from fastspeech.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser, device):
    dataloaders = {}
    config_params = list(configs["data"].items())
    for i in range(len(config_params)):
        assert config_params[i][0] == "all", "Data type must be one all"
        assert len(config_params) == 1, "With all specified -- use only" \
                                        " one dataset"
        params = config_params[i][1]
        num_workers = params.get("num_workers", 1)
        dataset = configs.init_obj(params["datasets"][0], fastspeech.datasets,
                                   device, num_workers, configs)
        assert "test_share" in params, "You must specify share of test " \
                                       "examples"
        test_share = float(params["test_share"])
        test_size = int(test_share * len(dataset))
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])
        # select batch size or batch sampler
        assert "batch_size" in params,\
            "You must provide batch_size for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
        else:
            raise Exception()
        train_dataloader = DataLoader(
            train_dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers)
        test_dataloader = DataLoader(
            test_dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers)
        dataloaders["train"] = train_dataloader
        dataloaders["val"] = test_dataloader
    return dataloaders
