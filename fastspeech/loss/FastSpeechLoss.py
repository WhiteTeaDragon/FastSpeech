import torch


class FastSpeechLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> dict:
        slice_until = min(kwargs["output_duration"].shape[-1],
                          kwargs["duration"].shape[-1])
        duration_loss = torch.nn.functional.mse_loss(torch.exp(
            kwargs["output_duration"][..., :slice_until]),
            kwargs["duration"][..., :slice_until].float())
        slice_until = min(kwargs["output_melspec"].shape[-1],
                          kwargs["melspec"].shape[-1])
        mask = ~kwargs["output_mask"].detach()
        output_term = (kwargs["output_melspec"][..., :slice_until].transpose(0,
                                                                             1)
                       * mask[..., :slice_until]).transpose(0, 1)
        true_term = (kwargs["melspec"][..., :slice_until].transpose(0, 1) *
                     mask[..., :slice_until]).transpose(0, 1)
        melspec_loss = torch.nn.functional.mse_loss(output_term, true_term)
        return {"loss": duration_loss + melspec_loss,
                "duration_loss": duration_loss, "melspec_loss": melspec_loss}
