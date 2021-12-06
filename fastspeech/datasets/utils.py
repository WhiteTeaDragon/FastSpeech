import re

import torch
from torch.utils.data import DataLoader

import fastspeech.datasets
from fastspeech.collate_fn.collate import collate_fn
from fastspeech.utils.parse_config import ConfigParser
from unidecode import unidecode


def get_dataloaders(configs: ConfigParser, device):
    dataloaders = {}
    config_params = list(configs["data"].items())
    for i in range(len(config_params)):
        assert config_params[i][0] == "all" or config_params[i][0] == "test", \
            "Data type must be one all or one test"
        assert len(config_params) == 1, "With all specified -- use only" \
                                        " one dataset"
        params = config_params[i][1]
        num_workers = params.get("num_workers", 1)
        dataset = configs.init_obj(params["datasets"][0], fastspeech.datasets,
                                   device, num_workers, configs)
        if "test_size" in params:
            test_size = int(params["test_size"])
            train_size = len(dataset) - test_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, test_size])
            split = "train"
        else:
            train_dataset = dataset
            split = "test"
        # select batch size or batch sampler
        assert "batch_size" in params,\
            "You must provide batch_size for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = False
        else:
            raise Exception()
        train_dataloader = DataLoader(
            train_dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers)
        dataloaders[split] = train_dataloader
        if "test_size" in params:
            test_dataloader = DataLoader(
                test_dataset, batch_size=bs, collate_fn=collate_fn,
                shuffle=shuffle, num_workers=num_workers)
            dataloaders["val"] = test_dataloader
    return dataloaders


# from https://github.com/xcmyz/FastSpeech/blob/master/text/cleaners.py
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in
                  [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def collapse_whitespace(text):
    return re.sub(re.compile(r'\s+'), ' ', text)


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, unidecode(text))
    return collapse_whitespace(text)
