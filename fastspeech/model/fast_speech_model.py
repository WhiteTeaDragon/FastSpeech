import math

from torch import nn
import torchaudio
import torch
from fastspeech.utils import ROOT_PATH
from google_drive_downloader import GoogleDriveDownloader as gdd

from fastspeech.base import BaseModel


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, emb_size):
        super(MultiHeadedAttention, self).__init__()
        self.n_heads = n_heads
        self.hidden_size = emb_size
        self.queries, self.keys, self.values = [], [], []
        for i in range(n_heads):
            self.queries.append(nn.Linear(emb_size, emb_size))
            self.keys.append(nn.Linear(emb_size, emb_size))
            self.values.append(nn.Linear(emb_size, emb_size))
        self.w_final = nn.Linear(emb_size * n_heads, emb_size)

    def forward(self, inputs):
        final_res = None
        for i in range(self.n_heads):
            q = self.queries[i](inputs)
            k = self.keys[i](inputs)
            v = self.values[i](inputs)
            res = torch.matmul(q, k.t()) / math.sqrt(self.hidden_size)
            res = torch.matmul(torch.nn.functional.softmax(res, dim=1), v)
            if final_res is None:
                final_res = res
            else:
                final_res = torch.cat((final_res, res), dim=1)
        return self.w_final(final_res)


class FeedForwardTransformer(nn.Module):
    def __init__(self, n_heads, emb_size, kernel_size):
        super(FeedForwardTransformer, self).__init__()
        self.attention = MultiHeadedAttention(n_heads, emb_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.conv = nn.Conv1d(emb_size, emb_size, kernel_size)
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, inputs):
        after_attention = self.attention(inputs)
        inputs = after_attention + inputs
        inputs = self.norm1(inputs)
        after_conv = self.conv(inputs)
        inputs = after_conv + inputs
        inputs = self.norm2(inputs)
        return inputs


class DurationPredictor(nn.Module):
    def __init__(self, emb_size, hidden_size, kernel_size, alpha):
        super(DurationPredictor, self).__init__()
        self.conv1 = nn.Conv1d(emb_size, hidden_size, kernel_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.conv2 = nn.Conv1d(hidden_size, emb_size, kernel_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, 1)
        self.alpha = alpha

    def forward(self, inputs):
        inputs = self.norm1(self.conv1(inputs))
        inputs = self.norm2(self.conv2(inputs))
        return self.linear(inputs) * self.alpha


def length_regulation(inputs, durations):
    final_res = None
    batch, seq_len = inputs.shape
    for i in range(batch):
        curr_element = None
        for j in range(seq_len):
            curr_res = inputs[i, j].repeat(1, durations[j])
            if curr_element is None:
                curr_element = curr_res
            else:
                curr_element = torch.cat((curr_element, curr_res), dim=1)
        if final_res is None:
            final_res = curr_element
        else:
            final_res = torch.cat((final_res, curr_element))
    return final_res


class Vocoder(nn.Module):
    def __init__(self):
        super(Vocoder, self).__init__()
        data_dir = ROOT_PATH / "vocoder" / \
            "waveglow_256channels_universal_v5.pt"
        if not data_dir.exists():
            gdd.download_file_from_google_drive(
                file_id='1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF',
                dest_path=data_dir
            )
        model = torch.load(data_dir, map_location='cpu')['model']
        self.net = model.remove_weightnorm(model)

    @torch.no_grad()
    def inference(self, spect: torch.Tensor):
        spect = self.net.upsample(spect)

        # trim the conv artifacts
        time_cutoff = self.net.upsample.kernel_size[0] - \
                      self.net.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.net.n_group, self.net.n_group) \
            .permute(0, 2, 1, 3) \
            .contiguous() \
            .flatten(start_dim=2) \
            .transpose(-1, -2)

        # generate prior
        audio = torch.randn(spect.size(0), self.net.n_remaining_channels,
                            spect.size(-1)).to(spect.device)

        for k in reversed(range(self.net.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.net.WN[k]((audio_0, spect))

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.net.convinv[k](audio, reverse=True)

            if k % self.net.n_early_every == 0 and k > 0:
                z = torch.randn(
                    spect.size(0), self.net.n_early_size, spect.size(2),
                    device=spect.device
                )
                audio = torch.cat((z, audio), 1)

        audio = audio.permute(0, 2, 1) \
            .contiguous() \
            .view(audio.size(0), -1)

        return audio


class FastSpeechModel(BaseModel):
    def __init__(self, emb_size, max_len, num_blocks, n_heads, kernel_size,
                 predictor_hidden_size, predictor_kernel_size, alpha, mels,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        vocab_size = len(torchaudio.pipelines.
                         WAV2VEC2_ASR_BASE_960H.get_labels())
        self.embedding = nn.Embedding(vocab_size, emb_size)
        w = torch.Tensor([(1 / 10000) ** (2 * i / emb_size) for i in
                          range(1, emb_size // 2 + 1)])
        position_ids = torch.arange(max_len).unsqueeze(1)
        multiplied = w * position_ids
        pos_enc = torch.zeros(max_len, emb_size)
        pos_enc[:, ::2] = torch.sin(multiplied)
        pos_enc[:, 1::2] = torch.cos(multiplied)
        self.register_buffer("pos_enc", pos_enc)
        blocks = []
        for i in range(num_blocks):
            blocks.append(FeedForwardTransformer(n_heads, emb_size,
                                                 kernel_size))
        self.fft1 = nn.Sequential(*blocks)
        self.duration_predictor = DurationPredictor(emb_size,
                                                    predictor_hidden_size,
                                                    predictor_kernel_size,
                                                    alpha)
        blocks = []
        for i in range(num_blocks):
            blocks.append(FeedForwardTransformer(n_heads, emb_size,
                                                 kernel_size))
        self.fft2 = nn.Sequential(*blocks)
        self.linear = nn.Linear(emb_size, mels)
        self.vocoder = Vocoder()

    def forward(self, inputs, durations=None, *args, **kwargs):
        inputs = self.embedding(inputs)
        batch, seq_len = inputs.shape
        inputs = inputs + self.pos_enc[:seq_len]
        inputs = self.fft1(inputs)
        duration_prediction = self.duration_predictor(inputs)
        if durations is None:
            inputs = length_regulation(inputs,
                                       torch.exp(duration_prediction))
        else:
            inputs = length_regulation(inputs, durations)
        batch, seq_len = inputs.shape
        inputs = inputs + self.pos_enc[:seq_len]
        inputs = self.fft2(inputs)
        spectrogram = self.linear(inputs)
        audio = self.vocoder.inference(spectrogram)
        return {"output_melspec": spectrogram,
                "output_duration": duration_prediction,
                "output_audio": audio}
