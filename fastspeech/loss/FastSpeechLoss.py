import torch

from fastspeech.collate_fn.collate import build_mask


class FastSpeechLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> dict:
        mask = kwargs["mask"]
        log_duration = torch.log(kwargs["duration"])
        log_duration[mask] = 0
        duration_loss = torch.nn.functional.mse_loss(kwargs["output_duration"].
                                                     squeeze(-1) * mask,
                                                     log_duration)
        slice_until = min(kwargs["output_melspec"].shape[-1],
                          kwargs["melspec"].shape[-1])
        melspec_lengths = kwargs["melspec_lengths"]
        melspec_lengths = torch.maximum(melspec_lengths, slice_until)
        mask = build_mask(melspec_lengths)
        melspec_loss = torch.nn.functional.mse_loss(kwargs["output_melspec"][
                                                    ..., :slice_until] * mask,
                                                    kwargs["melspec"][...,
                                                    :slice_until] * mask)
        return {"loss": duration_loss + melspec_loss,
                "duration_loss": duration_loss, "melspec_loss": melspec_loss}
