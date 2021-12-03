import logging
import shutil

import librosa
import numpy as np
import torchaudio
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from speechbrain.utils.data_utils import download_file

from fastspeech.collate_fn.collate import collate_fn
from fastspeech.datasets.GraphemeAligner import GraphemeAligner
from fastspeech.datasets.utils import expand_abbreviations
from fastspeech.utils import ROOT_PATH

logger = logging.getLogger(__name__)


def load_durations_from_outside(data_dir):
    zip_alignments_file = data_dir / "alignments.zip"
    if not zip_alignments_file.exists():
        download_file("https://github.com/xcmyz/FastSpeech/raw/master"
                      "/alignments.zip", zip_alignments_file)
    shutil.unpack_archive(zip_alignments_file, data_dir)
    durations = {}
    for fpath in (data_dir / "alignments").iterdir():
        filename = str(data_dir / "alignments" / fpath.name)
        index = int(fpath.name[:-4])
        durations[index] = torch.from_numpy(np.load(filename))
    res = []
    for i in range(len(durations)):
        res.append(durations[i])
    return res


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, device, num_workers, config_parser, aligner_bs,
                 durations_from_outside, data_dir=None, download="True"):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "lj"
            data_dir.mkdir(exist_ok=True, parents=True)
        super().__init__(root=data_dir, download=(download == "True"))
        self.config_parser = config_parser
        self._tokenizer = torchaudio.pipelines. \
            TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self.durations = None
        self.durations_from_outside = (durations_from_outside == "True")
        if self.durations_from_outside:
            self.durations = load_durations_from_outside(data_dir)
        else:
            self.durations = self.load_durations(device, num_workers,
                                                 config_parser, aligner_bs)

    def load_durations(self, device, num_workers, config_parser, aligner_bs):
        durations_dir = ROOT_PATH / "data" / "datasets" / "lj"
        durations_file = durations_dir / "durations.npy"
        if durations_file.exists():
            durations = torch.from_numpy(np.load(durations_file))
        else:
            durations_dir.mkdir(exist_ok=True, parents=True)
            aligner = GraphemeAligner(config_parser).to(device)
            dataloader = DataLoader(self, batch_size=aligner_bs,
                                    collate_fn=collate_fn, shuffle=False,
                                    num_workers=num_workers)
            if config_parser["overfit_on_one_batch"] == "True":
                dataloader = [next(iter(dataloader))]
            len_epoch = len(dataloader)
            durations = []
            for batch_idx, batch in enumerate(
                    tqdm(dataloader, desc="graphemes", total=len_epoch)
            ):
                correct_text = []
                for i in range(len(batch["text_encoded"])):
                    input = batch["text_encoded"][i]
                    length = batch["token_lengths"][i]
                    correct_text.append(''.join([self._tokenizer.tokens[j] for
                                                 j in input[:length]]))
                with torch.no_grad():
                    curr_durations = aligner(
                        batch["audio"].to(device), batch["audio_length"],
                        correct_text
                    )
                    hop_length = self.config_parser["preprocessing"][
                        "spectrogram"]["args"]["hop_length"]
                    coeff = batch["audio_length"] / hop_length
                    curr_durations *= coeff.repeat(curr_durations.shape[-1],
                                                   1).transpose(0, 1)
                durations += list(curr_durations)
            durations = pad_sequence(durations)
            durations = durations.reshape(durations.shape[0], -1).transpose(0,
                                                                            1)
            np.save(durations_file, durations)
        return durations

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        if self.durations_from_outside:
            transcript = expand_abbreviations(transcript)
        tokens, token_lengths = self._tokenizer(transcript)

        duration = None
        if self.durations is not None:
            duration = self.durations[index].unsqueeze(0)

        return waveform, waveform_length, transcript, tokens, token_lengths,\
               duration

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
