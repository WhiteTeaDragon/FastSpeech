import math

from torch import nn
import torchaudio
import torch

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
        self.queries = nn.ModuleList(self.queries)
        self.keys = nn.ModuleList(self.keys)
        self.values = nn.ModuleList(self.values)
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
    def __init__(self, n_heads, emb_size, hidden_size, kernel_size,
                 dropout_p):
        super(FeedForwardTransformer, self).__init__()
        self.attention = MultiHeadedAttention(n_heads, emb_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_size, hidden_size, kernel_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, emb_size, kernel_size),
            nn.Dropout(p=dropout_p)
        )
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


class FastSpeechModel(BaseModel):
    def __init__(self, emb_size, max_len, num_blocks, n_heads, kernel_size,
                 fft_hidden_size, dropout_p, predictor_kernel_size, alpha,
                 mels, *args, **kwargs):
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
                                                 fft_hidden_size, kernel_size,
                                                 dropout_p))
        self.fft1 = nn.Sequential(*blocks)
        self.duration_predictor = DurationPredictor(emb_size,
                                                    emb_size,
                                                    predictor_kernel_size,
                                                    alpha)
        blocks = []
        for i in range(num_blocks):
            blocks.append(FeedForwardTransformer(n_heads, emb_size,
                                                 fft_hidden_size, kernel_size,
                                                 dropout_p))
        self.fft2 = nn.Sequential(*blocks)
        self.linear = nn.Linear(emb_size, mels)

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
        return {"output_melspec": spectrogram,
                "output_duration": duration_prediction}
