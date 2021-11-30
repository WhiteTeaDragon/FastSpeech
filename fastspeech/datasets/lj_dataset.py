import logging
import torchaudio
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .GraphemeAligner import GraphemeAligner
from fastspeech.utils import ROOT_PATH
from ..collate_fn.collate import collate_fn

logger = logging.getLogger(__name__)


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, device, num_workers, config_parser, data_dir=None):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "lj"
            data_dir.mkdir(exist_ok=True, parents=True)
        super().__init__(root=data_dir, download=True)
        self.config_parser = config_parser
        self._tokenizer = torchaudio.pipelines. \
            TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self.durations = None
        self.durations = self.load_durations(data_dir, device, num_workers,
                                             config_parser)

    def load_durations(self, data_dir, device, num_workers, config_parser):
        durations_file = data_dir / "durations.npy"
        if durations_file.exists():
            durations = torch.from_numpy(np.load(durations_file))
        else:
            aligner = GraphemeAligner(config_parser).to(device)
            dataloader = DataLoader(self, batch_size=4, collate_fn=collate_fn,
                                    shuffle=False, num_workers=num_workers)
            len_epoch = len(dataloader)
            durations = None
            for batch_idx, batch in enumerate(
                    tqdm(dataloader, desc="graphemes", total=len_epoch)
            ):
                with torch.no_grad():
                    curr_durations = aligner(
                        batch["audio"].to(device), batch["audio_length"],
                        batch["text"]
                    )
                if durations is None:
                    durations = curr_durations
                else:
                    durations = torch.cat((durations, curr_durations))
            np.save(durations_file, durations)
        return durations

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        audio_spec, sr = self.get_spectogram(waveform)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        tokens, token_lengths = self._tokenizer(transcript)
        duration = None
        if self.durations is not None:
            duration = self.durations[index]

        return waveform, waveform_length, transcript, tokens, token_lengths, \
            duration, audio_spec, sr

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result

    def get_spectogram(self, audio_tensor_wave: torch.Tensor):
        sr = self.config_parser["preprocessing"]["sr"]
        with torch.no_grad():
            wave2spec = self.config_parser.init_obj(
                self.config_parser["preprocessing"]["spectrogram"],
                torchaudio.transforms,
            )
            audio_tensor_spec = wave2spec(audio_tensor_wave)
        return audio_tensor_spec, sr
