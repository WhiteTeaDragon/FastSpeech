from torch.utils.data import DataLoader, ChainDataset

import fastspeech.datasets
from fastspeech.collate_fn.collate import collate_fn
from fastspeech.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser, device):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(ds, fastspeech.datasets, device,
                                             num_workers, configs))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ChainDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert "batch_size" in params,\
            "You must provide batch_size for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
        else:
            raise Exception()

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers)
        dataloaders[split] = dataloader
    return dataloaders
