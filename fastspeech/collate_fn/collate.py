import logging
from typing import List, Tuple, Dict
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(instances: List[Tuple]) -> Dict:
    """
    Collate and pad fields in dataset items
    """
    waveform, waveform_length, transcript, tokens, token_lengths, duration, \
        audio_spec, sr = list(zip(*instances))

    waveform = pad_sequence([
        waveform_[0] for waveform_ in waveform
    ]).transpose(0, 1)
    waveform_length = torch.cat(waveform_length)

    tokens = pad_sequence([
        tokens_[0] for tokens_ in tokens
    ]).transpose(0, 1)
    token_lengths = torch.cat(token_lengths)

    duration = pad_sequence([
        duration_[0] for duration_ in duration
    ]).transpose(0, 1)
    duration = torch.cat(duration)

    return {"audio": waveform, "audio_length": waveform_length,
            "text": transcript, "text_encoded": tokens,
            "token_lengths": token_lengths, "duration": duration,
            "melspec": audio_spec, "sample_rate": sr}