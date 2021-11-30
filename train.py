import argparse
import collections
import warnings

import numpy as np
import torch

import fastspeech.loss as module_loss
import fastspeech.model as module_arch
from fastspeech.datasets.utils import get_dataloaders
from fastspeech.trainer import Trainer
from fastspeech.utils import prepare_device
from fastspeech.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    device, device_ids = prepare_device(config["n_gpu"])
    # setup data_loader instances
    dataloaders, text_encoder = get_dataloaders(config, device)
    if config["overfit_on_one_batch"] == "True":
        dataloaders["train"] = [next(iter(dataloaders["train"]))]

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch,
                            n_class=len(text_encoder))
    logger.info(model)

    # prepare for (multi-device) GPU training
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = []

    # build optimizer, learning rate scheduler. delete every lines
    # containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim,
                                trainable_params)
    lr_scheduler = config.init_obj(config["lr_scheduler"],
                                   torch.optim.lr_scheduler, optimizer)
    scheduler_frequency_of_update = config["lr_scheduler"]["frequency"]
    do_beam_search = config["trainer"].get("beam_search", False)
    if do_beam_search == "True":
        do_beam_search = True
    else:
        do_beam_search = False

    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer,
        text_encoder=text_encoder,
        config=config,
        device=device,
        data_loader=dataloaders["train"],
        valid_data_loader=dataloaders["val"],
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None),
        scheduler_frequency_of_update=scheduler_frequency_of_update,
        beam_search=do_beam_search
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given
    # in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float,
                   target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int,
            target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
