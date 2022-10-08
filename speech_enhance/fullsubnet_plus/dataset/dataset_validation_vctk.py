import os
from pathlib import Path

import librosa

from audio_zen.dataset.base_dataset import BaseDataset
from audio_zen.acoustics.feature import load_wav
from audio_zen.utils import basename


class Dataset(BaseDataset):
    def __init__(
            self,
            dataset_dir_list,
            sr,
    ):
        """
        Construct DNS validation set

        synthetic/
            with_reverb/
                noisy/
                clean_y/
            no_reverb/
                noisy/
                clean_y/
        """
        super(Dataset, self).__init__()
        with open(dataset_dir_list+'/clean.scp', "r") as f:
            dataset_list = {line.split(' ')[0] : {"clean":line.split(' ')[1].rstrip('\n')} for line in f.readlines()}

        with open(dataset_dir_list+'/noisy.scp', "r") as f:
            for line in f.readlines():
                id, noisypath = line.split(' ')
                dataset_list[id]['noisy'] = noisypath.rstrip('\n')

        self.length = len(dataset_list)
        self.noisy_files_list = dataset_list
        self.id_list = list(dataset_list.keys())
        self.length = len(self.id_list)
        self.sr = sr

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        """
        use the absolute path of the noisy speech to find the corresponding clean speech.

        Notes
            with_reverb and no_reverb dirs have same-named files.
            If we use `basename`, the problem will be raised (cover) in visualization.

        Returns:
            noisy: [waveform...], clean: [waveform...], type: [reverb|no_reverb] + name
        """
        id = self.id_list[item]
        noisy_file_path = self.noisy_files_list[id]['noisy']
        clean_file_path = self.noisy_files_list[id]['clean']

        noisy = load_wav(os.path.abspath(os.path.expanduser(noisy_file_path)), sr=self.sr)
        clean = load_wav(os.path.abspath(os.path.expanduser(clean_file_path)), sr=self.sr)

        return noisy, clean, id, 'No_reverb'
