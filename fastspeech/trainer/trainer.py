import random

import PIL
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from fastspeech.base import BaseTrainer
from fastspeech.datasets.GraphemeAligner import GraphemeAligner
from fastspeech.logger.utils import plot_spectrogram_to_buf
from fastspeech.utils import inf_loop, MetricTracker
from fastspeech.model import Vocoder


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            data_loader,
            valid_data_loader=None,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
            scheduler_frequency_of_update=None,
            beam_search=False
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.scheduler_frequency_of_update = scheduler_frequency_of_update
        self.beam_search = beam_search
        self.log_step = 10
        self.vocoder = Vocoder()
        self.aligner = GraphemeAligner().to(self.device)

        self.train_metrics = MetricTracker(
            "loss", "grad norm", "duration_loss", "melspec_loss",
            *[m.name for m in self.metrics],
            writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", "duration_loss", "melspec_loss",
            *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["audio", "audio_length", "text_encoded",
                               "token_lengths", "melspec"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(),
                self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", get_lr(self.optimizer)
                )
                self._log_one_prediction(batch["text"], batch["output_audio"],
                                         batch["sample_rate"])
                self._log_spectrogram(batch["melspec"],
                                      batch["output_melspec"])
                self._log_scalars(self.train_metrics)
                self._log_audio(batch["audio"], batch["sample_rate"], "train")
            if batch_idx >= self.len_epoch:
                break

        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None and \
                self.scheduler_frequency_of_update == "epoch":
            if isinstance(self.lr_scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                if not self.do_validation:
                    raise ValueError("Cannot use ReduceLROnPlateau if "
                                     "validation is off")
                self.lr_scheduler.step(val_log["loss"])
            else:
                self.lr_scheduler.step()

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        with torch.no_grad():
            duration = self.aligner(
                batch["audio"], batch["audio_length"], batch["text"]
            )
        batch["duration"] = duration
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        batch.update(outputs)
        with torch.no_grad():
            self.vocoder.eval()
            batch["output_audio"] = self.vocoder.inference(
                batch["output_melspec"])
        loss_dict = self.criterion(**batch)
        batch["loss"] = loss_dict["loss"]
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None and \
                    self.scheduler_frequency_of_update == "batch":
                self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        metrics.update("duration_loss", loss_dict["duration_loss"].item())
        metrics.update("melspec_loss", loss_dict["melspec_loss"].item())
        batch["beam_search"] = self.beam_search
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(self.valid_data_loader),
                    desc="validation",
                    total=len(self.valid_data_loader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.valid_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, "valid")
            self._log_scalars(self.valid_metrics)
            self._log_one_prediction(batch["text"], batch["output_audio"],
                                     batch["sample_rate"])
            self._log_spectrogram(batch["melspec"], batch["output_melspec"])
            self._log_audio(batch["audio"], batch["sample_rate"], part="val")

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="rice")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_one_prediction(self, text, output_audio, sample_rates):
        if self.writer is None:
            return
        log_index = torch.randint(low=0, high=len(text), size=(1,)).item()
        to_log_text = text[log_index]
        to_log_audio = output_audio[log_index]
        to_log_sample_rate = sample_rates[log_index]
        self.writer.add_text("text input", to_log_text)
        self.writer.add_audio("audio output", to_log_audio, to_log_sample_rate)

    def _log_spectrogram(self, target_spectrograms, spectrogram_batch):
        log_index = torch.randint(low=0, high=len(target_spectrograms),
                                  size=(1,)).item()
        target_spec = target_spectrograms[log_index]
        image = PIL.Image.open(
            plot_spectrogram_to_buf(target_spec.cpu().log()))
        self.writer.add_image("target spec", ToTensor()(image))
        output_spec = spectrogram_batch[log_index]
        image = PIL.Image.open(
            plot_spectrogram_to_buf(output_spec.cpu().log()))
        self.writer.add_image("output spec", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in
                 parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}",
                                   metric_tracker.avg(metric_name))

    def _log_audio(self, audio_batch, sample_rates, part):
        index = random.choice(range(len(audio_batch)))
        audio = audio_batch[index]
        sample_rate = sample_rates[index]
        self.writer.add_audio("audio target" + part, audio, sample_rate)
