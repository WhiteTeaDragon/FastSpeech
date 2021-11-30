import logging
from typing import List, Tuple, Dict
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from fastspeech.datasets.GraphemeAligner import GraphemeAligner

logger = logging.getLogger(__name__)


def collate_fn(instances: List[Tuple]) -> Dict:
    """
    Collate and pad fields in dataset items
    """
    waveform, waveform_length, transcript, tokens, token_lengths, \
        audio_spec, sr = list(zip(*instances))

    waveform = pad_sequence([
        waveform_[0] for waveform_ in waveform
    ]).transpose(0, 1)
    waveform_length = torch.cat(waveform_length)

    tokens = pad_sequence([
        tokens_[0] for tokens_ in tokens
    ]).transpose(0, 1)
    token_lengths = torch.cat(token_lengths)

    # if duration is not None and duration[0] is not None:
    #     duration = pad_sequence([
    #         duration_[0] for duration_ in duration
    #     ]).transpose(0, 1)
    #     duration = torch.cat(duration)

    return {"audio": waveform, "audio_length": waveform_length,
            "text": transcript, "text_encoded": tokens,
            "token_lengths": token_lengths,
            "melspec": audio_spec, "sample_rate": sr}
