import torch


class FastSpeechLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> dict:
        duration_loss = torch.nn.functional.mse_loss(kwargs["output_duration"],
                                                     torch.log(
                                                         kwargs["duration"]))
        slice_until = min(kwargs["output_melspec"].shape[-1],
            kwargs["melspec"].shape[-1])
        melspec_loss = torch.nn.functional.mse_loss(kwargs["output_melspec"][..., :slice_until],
                                                    kwargs["melspec"][..., :slice_until])
        return {"loss": duration_loss + melspec_loss,
                "duration_loss": duration_loss, "melspec_loss": melspec_loss}
