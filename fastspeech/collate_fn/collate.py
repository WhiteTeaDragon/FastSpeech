import logging
from typing import List, Tuple, Dict
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)
MELSPEC_PAD_VALUE = -11.5129251


def build_mask(token_lengths):
    max_len = token_lengths.max()
    mask = ~(torch.arange(max_len).expand(len(token_lengths),
                                          max_len) <
             token_lengths.unsqueeze(1))
    return mask


def collate_fn(instances: List[Tuple]) -> Dict:
    """
    Collate and pad fields in dataset items
    """
    waveform, waveform_length, transcript, tokens, token_lengths, \
        audio_spec, sr, duration = list(zip(*instances))

    waveform = pad_sequence([
        waveform_[0] for waveform_ in waveform
    ]).transpose(0, 1)
    waveform_length = torch.cat(waveform_length)

    tokens = pad_sequence([
        tokens_[0] for tokens_ in tokens
    ]).transpose(0, 1)
    token_lengths = torch.cat(token_lengths)

    if duration is not None and duration[0] is not None:
        duration = pad_sequence([
            duration_[0] for duration_ in duration
        ]).transpose(0, 1)

    melspec = pad_sequence([
        audio_spec_[0] for audio_spec_ in audio_spec
    ], padding_value=MELSPEC_PAD_VALUE).transpose(0, 1)
    melspec_lengths = torch.cat(token_lengths)

    mask = build_mask(token_lengths)

    return {"audio": waveform, "audio_length": waveform_length,
            "text": transcript, "text_encoded": tokens,
            "token_lengths": token_lengths, "melspec": melspec,
            "melspec_lengths": melspec_lengths, "sample_rate": sr,
            "duration": duration, "mask": mask}
