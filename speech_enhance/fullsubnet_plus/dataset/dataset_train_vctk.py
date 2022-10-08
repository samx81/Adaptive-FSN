import random

import numpy as np
from joblib import Parallel, delayed
from scipy import signal
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))  # without installation, add /path/to/Audio-ZEN

import torch
from audio_zen.dataset.base_dataset import BaseDataset
from fullsubnet_plus.dataset.pcs import PCS_Aug
from audio_zen.acoustics.feature import norm_amplitude, tailor_dB_FS, is_clipped, load_wav, subsample
from audio_zen.utils import expand_path


class Dataset(BaseDataset):
    def __init__(self,
                 clean_dataset,
                 noise_dataset,
                 sr,
                 num_workers,
                 pcs_aug=0,
                 clean_dataset_limit=None,
                 clean_dataset_offset=None,
                 noise_dataset_limit=None,
                 noise_dataset_offset=None,
                 rir_dataset=None,
                 rir_dataset_limit=None,
                 rir_dataset_offset=None,
                 snr_range=None,
                 reverb_proportion=None,
                 silence_length=None,
                 target_dB_FS=None,
                 target_dB_FS_floating_value=None,
                 sub_sample_length=None,
                 pre_load_clean_dataset=None,
                 pre_load_noise=None,
                 pre_load_rir=None,
                 ):
    # def __init__(self,
    #              clean_dataset,
    #              noise_dataset,
    #              sr,
    #              num_workers
    #              ):
        """
        Dynamic mixing for training

        Args:
            clean_dataset_limit:
            clean_dataset_offset:
            noise_dataset_limit:
            noise_dataset_offset:
            rir_dataset:
            rir_dataset_limit:
            rir_dataset_offset:
            snr_range:
            reverb_proportion:
            clean_dataset: scp file
            noise_dataset: scp file
            sub_sample_length:
            sr:
        """
        super().__init__()
        # acoustics args
        self.sr = sr

        # parallel args
        self.num_workers = num_workers

        # clean_dataset_list = {line.split(' ')[0] : {"clean":line.split(' ')[1]} for line in open(expand_path(clean_dataset), "r")}

        # noisy_dataset_list = {line.split(' ')[0] : line.split(' ')[1] for line in open(expand_path(noise_dataset), "r")}
        with open(expand_path(clean_dataset), "r") as f:
            dataset_list = {line.split(' ')[0] : {"clean":line.split(' ')[1].rstrip('\n')} for line in f.readlines()}
        with open(expand_path(noise_dataset), "r") as f:
            for line in f.readlines():
                id, noisypath = line.split(' ')
                dataset_list[id]['noisy'] = noisypath.rstrip('\n')

        self.dataset_list = dataset_list
        self.dataset_id = list(dataset_list.keys())
        self.pcs_aug = pcs_aug
        # self.noisy_dataset_list = noisy_dataset_list

        # snr_list = self._parse_snr_range(snr_range)
        # self.snr_list = snr_list

        # assert 0 <= reverb_proportion <= 1, "reverberation proportion should be in [0, 1]"
        # self.reverb_proportion = reverb_proportion
        # self.silence_length = silence_length
        # self.target_dB_FS = target_dB_FS
        # self.target_dB_FS_floating_value = target_dB_FS_floating_value
        self.sub_sample_length = sub_sample_length

        self.length = len(self.dataset_id)

    def __len__(self):
        return self.length

    @staticmethod
    def _random_select_from(dataset_list):
        return random.choice(dataset_list)

    def _select_noise_y(self, target_length):
        noise_y = np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(self.sr * self.silence_length), dtype=np.float32)
        remaining_length = target_length

        while remaining_length > 0:
            noise_file = self._random_select_from(self.noise_dataset_list)
            noise_new_added = load_wav(noise_file, sr=self.sr)
            noise_y = np.append(noise_y, noise_new_added)
            remaining_length -= len(noise_new_added)

            # 如果还需要添加新的噪声，就插入一个小静音段
            if remaining_length > 0:
                silence_len = min(remaining_length, len(silence))
                noise_y = np.append(noise_y, silence[:silence_len])
                remaining_length -= silence_len

        if len(noise_y) > target_length:
            idx_start = np.random.randint(len(noise_y) - target_length)
            noise_y = noise_y[idx_start:idx_start + target_length]

        return noise_y

    @staticmethod
    def snr_mix(clean_y, noise_y, snr, target_dB_FS, target_dB_FS_floating_value, rir=None, eps=1e-6):
        """
        混合噪声与纯净语音，当 rir 参数不为空时，对纯净语音施加混响效果

        Args:
            clean_y: 纯净语音
            noise_y: 噪声
            snr (int): 信噪比
            target_dB_FS (int):
            target_dB_FS_floating_value (int):
            rir: room impulse response, None 或 np.array
            eps: eps

        Returns:
            (noisy_y，clean_y)
        """
        if rir is not None:
            if rir.ndim > 1:
                rir_idx = np.random.randint(0, rir.shape[0])
                rir = rir[rir_idx, :]

            clean_y = signal.fftconvolve(clean_y, rir)[:len(clean_y)]

        clean_y, _ = norm_amplitude(clean_y)
        clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
        clean_rms = (clean_y ** 2).mean() ** 0.5

        noise_y, _ = norm_amplitude(noise_y)
        noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
        noise_rms = (noise_y ** 2).mean() ** 0.5

        snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y + noise_y

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value
        )

        # 使用 noisy 的 rms 放缩音频
        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar

        # 合成带噪语音的时候可能会 clipping，虽然极少
        # 对 noisy, clean_y, noise_y 稍微进行调整
        if is_clipped(noisy_y):
            noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)  # 相当于除以 1
            noisy_y = noisy_y / noisy_y_scalar
            clean_y = clean_y / noisy_y_scalar

        return noisy_y, clean_y

    def __getitem__(self, item):
        # clean_file = self.clean_dataset_list[item]
        id = self.dataset_id[item]
        clean_file = self.dataset_list[id]['clean']
        clean_y = load_wav(clean_file, sr=self.sr)
        # clean_y, start_position = subsample(clean_y, sub_sample_length=int(self.sub_sample_length * self.sr), return_start_position=True) # will randomly generate start of audio
        clean_y, start = subsample(clean_y, sub_sample_length=int(self.sub_sample_length * self.sr), return_start_position=True)
        noisy_file = self.dataset_list[id]['noisy']
        noisy_y = load_wav(noisy_file, sr=self.sr)
        noisy_y = subsample(noisy_y, sub_sample_length=int(self.sub_sample_length * self.sr),start_position=start)

        # noise_y = self._select_noise_y(target_length=len(clean_y))
        # assert len(clean_y) == len(noise_y), f"Inequality: {len(clean_y)} {len(noise_y)}"

        # snr = self._random_select_from(self.snr_list)
        # use_reverb = bool(np.random.random(1) < self.reverb_proportion)

        # noisy_y, clean_y = self.snr_mix(
        #     clean_y=clean_y,
        #     noise_y=noise_y,
        #     snr=snr,
        #     target_dB_FS=self.target_dB_FS,
        #     target_dB_FS_floating_value=self.target_dB_FS_floating_value,
        #     rir=load_wav(self._random_select_from(self.rir_dataset_list), sr=self.sr) if use_reverb else None
        # )

        noisy_y = noisy_y.astype(np.float32)

        if np.random.random(1) < self.pcs_aug:
        # pcs_aug == 0, then any number will be less then it
            noisy_y = PCS_Aug(torch.tensor(noisy_y)).numpy()

        clean_y = clean_y.astype(np.float32)

        return noisy_y, clean_y

class Aug_Dataset(BaseDataset):
    def __init__(self,
                 clean_dataset,
                 noise_dataset,
                 sr,
                 num_workers,
                 pcs_aug=0,
                 clean_dataset_limit=None,
                 clean_dataset_offset=None,
                 noise_dataset_limit=None,
                 noise_dataset_offset=None,
                 rir_dataset=None,
                 rir_dataset_limit=None,
                 rir_dataset_offset=None,
                 snr_range=None,
                 reverb_proportion=None,
                 silence_length=None,
                 target_dB_FS=None,
                 target_dB_FS_floating_value=None,
                 sub_sample_length=None,
                 pre_load_clean_dataset=None,
                 pre_load_noise=None,
                 pre_load_rir=None,
                 ):
        """
        Dynamic mixing for training

        Args:
            clean_dataset_limit:
            clean_dataset_offset:
            noise_dataset_limit:
            noise_dataset_offset:
            rir_dataset:
            rir_dataset_limit:
            rir_dataset_offset:
            snr_range:
            reverb_proportion:
            clean_dataset: scp file
            noise_dataset: scp file
            sub_sample_length:
            sr:
        """
        super().__init__()
        # acoustics args
        self.sr = sr

        # parallel args
        self.num_workers = num_workers

        with open(expand_path(clean_dataset), "r") as f:
            dataset_list = {line.split(' ')[0] : {"clean":line.split(' ')[1].rstrip('\n')} for line in f.readlines()}
        with open(expand_path(noise_dataset), "r") as f:
            for line in f.readlines():
                id, noisypath = line.split(' ')
                dataset_list[id]['noisy'] = noisypath.rstrip('\n')

        self.dataset_list = dataset_list
        self.dataset_id = list(dataset_list.keys())
        self.pcs_aug = pcs_aug
        self.sub_sample_length = sub_sample_length

        self.length = len(self.dataset_id)

    def __len__(self):
        return self.length

    @staticmethod
    def _random_select_from(dataset_list):
        return random.choice(dataset_list)

    def _select_noise_y(self, target_length):
        noise_y = np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(self.sr * self.silence_length), dtype=np.float32)
        remaining_length = target_length

        while remaining_length > 0:
            noise_file = self._random_select_from(self.noise_dataset_list)
            noise_new_added = load_wav(noise_file, sr=self.sr)
            noise_y = np.append(noise_y, noise_new_added)
            remaining_length -= len(noise_new_added)

            # 如果还需要添加新的噪声，就插入一个小静音段
            if remaining_length > 0:
                silence_len = min(remaining_length, len(silence))
                noise_y = np.append(noise_y, silence[:silence_len])
                remaining_length -= silence_len

        if len(noise_y) > target_length:
            idx_start = np.random.randint(len(noise_y) - target_length)
            noise_y = noise_y[idx_start:idx_start + target_length]

        return noise_y

    @staticmethod
    def snr_mix(clean_y, noise_y, snr, target_dB_FS, target_dB_FS_floating_value, rir=None, eps=1e-6):
        """
        混合噪声与纯净语音，当 rir 参数不为空时，对纯净语音施加混响效果

        Args:
            clean_y: 纯净语音
            noise_y: 噪声
            snr (int): 信噪比
            target_dB_FS (int):
            target_dB_FS_floating_value (int):
            rir: room impulse response, None 或 np.array
            eps: eps

        Returns:
            (noisy_y，clean_y)
        """
        if rir is not None:
            if rir.ndim > 1:
                rir_idx = np.random.randint(0, rir.shape[0])
                rir = rir[rir_idx, :]

            clean_y = signal.fftconvolve(clean_y, rir)[:len(clean_y)]

        clean_y, _ = norm_amplitude(clean_y)
        clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
        clean_rms = (clean_y ** 2).mean() ** 0.5

        noise_y, _ = norm_amplitude(noise_y)
        noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
        noise_rms = (noise_y ** 2).mean() ** 0.5

        snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y + noise_y

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value
        )

        # 使用 noisy 的 rms 放缩音频
        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar

        # 合成带噪语音的时候可能会 clipping，虽然极少
        # 对 noisy, clean_y, noise_y 稍微进行调整
        if is_clipped(noisy_y):
            noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)  # 相当于除以 1
            noisy_y = noisy_y / noisy_y_scalar
            clean_y = clean_y / noisy_y_scalar

        return noisy_y, clean_y

    def __getitem__(self, item):
        # clean_file = self.clean_dataset_list[item]
        id = self.dataset_id[item]
        clean_file = self.dataset_list[id]['clean']
        clean_y = load_wav(clean_file, sr=self.sr)
        # clean_y, start_position = subsample(clean_y, sub_sample_length=int(self.sub_sample_length * self.sr), return_start_position=True) # will randomly generate start of audio
        clean_y, start = subsample(clean_y, sub_sample_length=int(self.sub_sample_length * self.sr), return_start_position=True)
        noisy_file = self.dataset_list[id]['noisy']
        noisy_y = load_wav(noisy_file, sr=self.sr)
        noisy_y = subsample(noisy_y, sub_sample_length=int(self.sub_sample_length * self.sr),start_position=start)

        noisy_y = noisy_y.astype(np.float32)

        aug_noisy_y = PCS_Aug(torch.tensor(noisy_y)).numpy()
        aug_clean_y = PCS_Aug(torch.tensor(clean_y)).numpy()

        clean_y = clean_y.astype(np.float32)

        return aug_noisy_y, noisy_y, clean_y, aug_clean_y


if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))  # without installation, add /path/to/Audio-ZEN
    dset = Dataset(
        clean_dataset = "/share/nas167/samtsao/pj/enhance/FullSubNet/recipes/vctk/data/voicebank/cv/clean.scp",
        noise_dataset = "/share/nas167/samtsao/pj/enhance/FullSubNet/recipes/vctk/data/voicebank/cv/noisy.scp",
        sr = 16000,
        num_workers = 36)
    print(len(dset))
