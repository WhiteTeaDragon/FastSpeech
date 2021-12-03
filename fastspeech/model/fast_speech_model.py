import math

from torch import nn
import torch
import torchaudio

from fastspeech.base import BaseModel


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, emb_size, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert emb_size % n_heads == 0, f"Wrong number of heads {n_heads} " \
                                        f"for emb_size {emb_size}!"
        self.n_heads = n_heads
        self.hidden_size = emb_size
        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.w_final = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def reshape_into_heads(self, out, batch_size):
        last_dim = self.hidden_size // self.n_heads
        out = out.reshape(batch_size, -1, self.n_heads, last_dim)
        return torch.permute(out, (0, 2, 1, 3))

    def forward(self, inputs, mask=None):
        batch_size = inputs.shape[0]
        q = self.reshape_into_heads(self.query(inputs), batch_size)
        k = self.reshape_into_heads(self.key(inputs), batch_size)
        v = self.reshape_into_heads(self.value(inputs), batch_size)
        res = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
            self.hidden_size)
        if mask is not None:
            res = res.permute(1, 3, 0, 2)
            res += mask * 1e-9
            res = res.permute(2, 0, 3, 1)
        att = torch.nn.functional.softmax(res, dim=2)
        res = torch.matmul(att, v)
        res = res.permute(0, 2, 1, 3).contiguous().view(batch_size, -1,
                                                        self.hidden_size)
        return self.dropout(self.w_final(res)), att


class FeedForwardTransformer(nn.Module):
    def __init__(self, n_heads, emb_size, hidden_size, kernel_size,
                 dropout_p):
        super(FeedForwardTransformer, self).__init__()
        self.attention = MultiHeadedAttention(n_heads, emb_size, dropout_p)
        self.norm1 = nn.LayerNorm(emb_size)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_size, hidden_size, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv1d(hidden_size, emb_size, kernel_size, padding="same"),
            nn.Dropout(p=dropout_p)
        )
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, inputs, mask):
        after_attention, att = self.attention(inputs, mask)
        inputs = after_attention + inputs
        inputs = self.norm1(inputs)
        after_conv = self.conv(inputs.transpose(1, 2)).transpose(1, 2)
        inputs = after_conv + inputs
        inputs = self.norm2(inputs)
        return inputs, att


class DurationPredictor(nn.Module):
    def __init__(self, emb_size, hidden_size, kernel_size, dropout):
        super(DurationPredictor, self).__init__()
        self.conv1 = nn.Conv1d(emb_size, hidden_size, kernel_size,
                               padding="same")
        self.norm1 = nn.LayerNorm(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size,
                               padding="same")
        self.norm2 = nn.LayerNorm(hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        inputs = self.norm1(self.conv1(inputs.transpose(1, 2)).transpose(1, 2))
        inputs = self.dropout1(self.relu1(inputs))
        inputs = self.norm2(self.conv2(inputs.transpose(1, 2)).transpose(1, 2))
        inputs = self.dropout2(self.relu2(inputs))
        return self.linear(inputs).squeeze(-1)


def length_regulation(inputs, durations, device):
    batch, seq_len, emb_size = inputs.shape
    true_lens = durations.sum(-1)
    true_len = round(true_lens.max().item())
    mask = torch.zeros(batch, true_len, device=device, dtype=bool)
    final_res = torch.zeros(batch, true_len, emb_size, device=device)
    for i in range(batch):
        index = 0
        for j in range(seq_len):
            if durations.dtype != torch.int:
                curr_len = torch.round(durations[i, j]).int().item()
            else:
                curr_len = durations[i, j].item()
            final_res[i, index:index + curr_len] = inputs[i, j]
            index += curr_len
        mask[i, round(true_lens[i].item()):] = True
    return final_res, mask


class FastSpeechModel(BaseModel):
    def __init__(self, emb_size, max_len, num_blocks, n_heads, kernel_size,
                 fft_hidden_size, dropout_p, predictor_hidden_size,
                 predictor_kernel_size, mels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vocab_size = len(torchaudio.pipelines.
                         TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.
                         get_text_processor().tokens)
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
        self.fft1 = nn.ModuleList(blocks)
        self.duration_predictor = DurationPredictor(emb_size,
                                                    predictor_hidden_size,
                                                    predictor_kernel_size,
                                                    dropout_p)
        blocks = []
        for i in range(num_blocks):
            blocks.append(FeedForwardTransformer(n_heads, emb_size,
                                                 fft_hidden_size, kernel_size,
                                                 dropout_p))
        self.fft2 = nn.ModuleList(blocks)
        self.linear = nn.Linear(emb_size, mels)

    def forward(self, text_encoded, token_lengths, device, mask, duration=None,
                alpha=1, *args, **kwargs):
        inputs = self.embedding(text_encoded)
        batch, seq_len, emb_size = inputs.shape
        inputs = inputs + self.pos_enc[:seq_len]
        attention = []
        for i in range(len(self.fft1)):
            inputs, att = self.fft1[i](inputs, mask)
            attention.append(att)
        duration_prediction = self.duration_predictor(inputs)
        if not self.training:
            inputs, mask = length_regulation(inputs,
                                             torch.exp(duration_prediction) *
                                             alpha,
                                             device)
        else:
            inputs, mask = length_regulation(inputs, duration * alpha,
                                             device)
        batch, seq_len, emb_size = inputs.shape
        inputs = inputs + self.pos_enc[:seq_len]
        for i in range(len(self.fft2)):
            inputs, att = self.fft2[i](inputs, mask)
            attention.append(att)
        spectrogram = self.linear(inputs)
        return {"output_melspec": spectrogram.transpose(1, 2),
                "output_duration": duration_prediction,
                "output_mask": mask, "attention": attention}
