import logging
import torchaudio
from torch.utils.data import Dataset

from fastspeech.datasets.utils import expand_abbreviations

logger = logging.getLogger(__name__)


class TestDataset(Dataset):
    def __init__(self, device, num_workers, config_parser, data_dir=None,
                 download="True", *args, **kwargs):
        super().__init__()
        self.config_parser = config_parser
        self._tokenizer = torchaudio.pipelines. \
            TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self.texts = [
            "A defibrillator is a device that gives a high energy electric "
            "shock to the heart of someone who is in cardiac arrest",
            "Massachusetts Institute of Technology may be best known for its "
            "math, science and engineering education",
            "Wasserstein distance or Kantorovich Rubinstein metric is a "
            "distance function defined between probability distributions on a "
            "given metric space"
        ]

    def __getitem__(self, index: int):
        transcript = self.texts[index]
        transcript = expand_abbreviations(transcript)
        tokens, token_lengths = self._tokenizer(transcript)

        return transcript, tokens, token_lengths

    def __len__(self):
        return len(self.texts)
