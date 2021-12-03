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
                                                     squeeze(-1) * (~mask),
                                                     log_duration)
        slice_until = min(kwargs["output_melspec"].shape[-1],
                          kwargs["melspec"].shape[-1])
        mask = ~kwargs["output_mask"].detach()
        output_term = (kwargs["output_melspec"][..., :slice_until].transpose(0,
                                                                             1)
                       * mask).transpose(0, 1)
        true_term = (kwargs["melspec"][..., :slice_until].transpose(0, 1) *
                     mask).transpose(0, 1)
        melspec_loss = torch.nn.functional.mse_loss(output_term, true_term)
        return {"loss": duration_loss + melspec_loss,
                "duration_loss": duration_loss, "melspec_loss": melspec_loss}
