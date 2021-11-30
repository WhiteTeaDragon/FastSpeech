import torch


class FastSpeechLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> dict:
        duration_loss = torch.nn.functional.mse_loss(kwargs["output_duration"],
                                                     torch.log(
                                                         kwargs["duration"]))
        melspec_loss = torch.nn.functional.mse_loss(kwargs["output_melspec"],
                                                    kwargs["melspec"])
        return {"loss": duration_loss + melspec_loss,
                "duration_loss": duration_loss, "melspec_loss": melspec_loss}
