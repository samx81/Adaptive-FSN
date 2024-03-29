import torch
import time
import torch.nn.functional as F

import numpy as np

from audio_zen.acoustics.feature import mag_phase, drop_band
from audio_zen.acoustics.mask import decompress_cIRM
from audio_zen.inferencer.base_inferencer import BaseInferencer, tqdm

from functools import partial

# for log
from utils.logger import log
print=log

def cumulative_norm(input):
    eps = 1e-10
    device = input.device
    data_type = input.dtype
    n_dim = input.ndim

    assert n_dim in (3, 4)

    if n_dim == 3:
        n_channels = 1
        batch_size, n_freqs, n_frames = input.size()
    else:
        batch_size, n_channels, n_freqs, n_frames = input.size()
        input = input.reshape(batch_size * n_channels, n_freqs, n_frames)

    step_sum = torch.sum(input, dim=1)  # [B, T]
    step_pow_sum = torch.sum(torch.square(input), dim=1)

    cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]
    cumulative_pow_sum = torch.cumsum(step_pow_sum, dim=-1)  # [B, T]

    entry_count = torch.arange(n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device)
    entry_count = entry_count.reshape(1, n_frames)  # [1, T]
    entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

    cum_mean = cumulative_sum / entry_count  # B, T
    cum_var = (cumulative_pow_sum - 2 * cum_mean * cumulative_sum) / entry_count + cum_mean.pow(2)  # B, T
    cum_std = (cum_var + eps).sqrt()  # B, T

    cum_mean = cum_mean.reshape(batch_size * n_channels, 1, n_frames)
    cum_std = cum_std.reshape(batch_size * n_channels, 1, n_frames)

    x = (input - cum_mean) / cum_std

    if n_dim == 4:
        x = x.reshape(batch_size, n_channels, n_freqs, n_frames)

    return x

class Inferencer(BaseInferencer):
    def __init__(self, config, checkpoint_path, output_dir, file_mode=False):
        super().__init__(config, checkpoint_path, output_dir, file_mode)
        assert self.inference_config["type"] in dir(self), f"Not implemented Inferencer type: {self.inference_config['type']}"
        

    @torch.no_grad()
    def mag(self, noisy, inference_args):
        noisy_complex = self.torch_stft(noisy)
        noisy_mag, noisy_phase = mag_phase(noisy_complex)  # [B, F, T] => [B, 1, F, T]

        enhanced_mag = self.model(noisy_mag.unsqueeze(1)).squeeze(1)

        enhanced = self.torch_istft((enhanced_mag, noisy_phase), length=noisy.size(-1), use_mag_phase=True)
        enhanced = enhanced.detach().squeeze(0).cpu().numpy()

        return enhanced

    @torch.no_grad()
    def scaled_mask(self, noisy, inference_args):
        noisy_complex = self.torch_stft(noisy)
        noisy_mag, noisy_phase = mag_phase(noisy_complex)

        # [B, F, T] => [B, 1, F, T] => model => [B, 2, F, T] => [B, F, T, 2]
        noisy_mag = noisy_mag.unsqueeze(1)
        scaled_mask = self.model(noisy_mag)
        scaled_mask = scaled_mask.permute(0, 2, 3, 1)

        enhanced_complex = noisy_complex * scaled_mask
        enhanced = self.torch_istft(enhanced_complex, length=noisy.size(-1), use_mag_phase=False)
        enhanced = enhanced.detach().squeeze(0).cpu().numpy()

        return enhanced

    @torch.no_grad()
    def sub_band_crm_mask(self, noisy, inference_args):
        pad_mode = inference_args["pad_mode"]
        n_neighbor = inference_args["n_neighbor"]

        noisy = noisy.cpu().numpy().reshape(-1)
        noisy_D = self.librosa_stft(noisy)

        noisy_real = torch.tensor(noisy_D.real, device=self.device)
        noisy_imag = torch.tensor(noisy_D.imag, device=self.device)
        noisy_mag = torch.sqrt(torch.square(noisy_real) + torch.square(noisy_imag))  # [F, T]
        n_freqs, n_frames = noisy_mag.size()

        noisy_mag = noisy_mag.reshape(1, 1, n_freqs, n_frames)
        noisy_mag_padded = self._unfold(noisy_mag, pad_mode, n_neighbor)  # [B, N, C, F_s, T] <=> [1, 257, 1, 31, T]
        noisy_mag_padded = noisy_mag_padded.squeeze(0).squeeze(1)  # [257, 31, 200] <=> [B, F_s, T]

        pred_crm = self.model(noisy_mag_padded).detach()  # [B, 2, T] <=> [F, 2, T]
        pred_crm = pred_crm.permute(0, 2, 1).contiguous()  # [B, T, 2]

        lim = 9.99
        pred_crm = lim * (pred_crm >= lim) - lim * (pred_crm <= -lim) + pred_crm * (torch.abs(pred_crm) < lim)
        pred_crm = -10 * torch.log((10 - pred_crm) / (10 + pred_crm))

        enhanced_real = pred_crm[:, :, 0] * noisy_real - pred_crm[:, :, 1] * noisy_imag
        enhanced_imag = pred_crm[:, :, 1] * noisy_real + pred_crm[:, :, 0] * noisy_imag

        enhanced_real = enhanced_real.cpu().numpy()
        enhanced_imag = enhanced_imag.cpu().numpy()
        enhanced = self.librosa_istft(enhanced_real + 1j * enhanced_imag, length=len(noisy))
        return enhanced

    @torch.no_grad()
    def full_band_crm_mask(self, noisy, inference_args):
        noisy_complex = self.torch_stft(noisy)
        noisy_mag, _ = mag_phase(noisy_complex)

        noisy_mag = noisy_mag.unsqueeze(1)

        t1 = time.time()
        pred_crm = self.model(noisy_mag)
        t2 = time.time()

        pred_crm = pred_crm.permute(0, 2, 3, 1)

        pred_crm = decompress_cIRM(pred_crm)
        enhanced_real = pred_crm[..., 0] * noisy_complex.real - pred_crm[..., 1] * noisy_complex.imag
        enhanced_imag = pred_crm[..., 1] * noisy_complex.real + pred_crm[..., 0] * noisy_complex.imag
        enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)
        enhanced = self.torch_istft(enhanced_complex, length=noisy.size(-1))
        enhanced = enhanced.detach().squeeze(0).cpu().numpy()

        rtf = (t2 - t1) / (len(enhanced) * 1.0 / self.acoustic_config["sr"])

        return enhanced, rtf

    @torch.no_grad()
    def mag_complex_full_band_crm_mask(self, noisy, inference_args):
        noisy_complex = self.torch_stft(noisy)
        noisy_mag, _ = mag_phase(noisy_complex)

        noisy_mag = noisy_mag.unsqueeze(1)
        noisy_real = (noisy_complex.real).unsqueeze(1)
        noisy_imag = (noisy_complex.imag).unsqueeze(1)


        t1 = time.time()
        pred_crm = self.model(noisy_mag, noisy_real, noisy_imag)
        t2 = time.time()

        pred_crm = pred_crm.permute(0, 2, 3, 1)

        pred_crm = decompress_cIRM(pred_crm)
        enhanced_real = pred_crm[..., 0] * noisy_complex.real - pred_crm[..., 1] * noisy_complex.imag
        enhanced_imag = pred_crm[..., 1] * noisy_complex.real + pred_crm[..., 0] * noisy_complex.imag
      
        enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)
        enhanced = self.torch_istft(enhanced_complex, length=noisy.size(-1))
        enhanced = enhanced.detach().squeeze(0).cpu().numpy()

        #
        rtf = (t2 - t1) / (len(enhanced) * 1.0 / self.acoustic_config["sr"])
        tqdm.write(f"model rtf: {rtf}")

        return enhanced, rtf


    def time(self, noisy, inference_args):
        t1 = time.time()
        enhanced = self.model(noisy,False)
        t2 = time.time()

        enhanced = enhanced.detach().squeeze(0).squeeze(0).cpu().numpy()
        #
        rtf = (t2 - t1) / (len(enhanced) * 1.0 / self.acoustic_config["sr"])
        tqdm.write(f"model rtf: {rtf}")

        return enhanced, rtf

    @torch.no_grad()
    def complex_full_band_crm_mask(self, noisy, inference_args):
        noisy_complex = self.torch_stft(noisy)

        noisy_input = torch.stack([noisy_complex.real, noisy_complex.imag], dim=1)

        t1 = time.time()
        pred_crm = self.model(noisy_input)
        t2 = time.time()
        pred_crm = pred_crm.permute(0, 2, 3, 1)

        pred_crm = decompress_cIRM(pred_crm)
        enhanced_real = pred_crm[..., 0] * noisy_complex.real - pred_crm[..., 1] * noisy_complex.imag
        enhanced_imag = pred_crm[..., 1] * noisy_complex.real + pred_crm[..., 0] * noisy_complex.imag
        enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)
        enhanced = self.torch_istft(enhanced_complex, length=noisy.size(-1))
        enhanced = enhanced.detach().squeeze(0).cpu().numpy()

        #
        rtf = (t2 - t1) / (len(enhanced) * 1.0 / self.acoustic_config["sr"])

        return enhanced

    @torch.no_grad()
    def overlapped_chunk(self, noisy, inference_args):
        sr = self.acoustic_config["sr"]

        noisy = noisy.squeeze(0)

        num_mics = 8
        chunk_length = sr * inference_args["chunk_length"]
        chunk_hop_length = chunk_length // 2
        num_chunks = int(noisy.shape[-1] / chunk_hop_length) + 1

        win = torch.hann_window(chunk_length, device=noisy.device)

        prev = None
        enhanced = None
        # 模拟语音的静音段，防止一上来就给语音，处理的不好
        for chunk_idx in range(num_chunks):
            if chunk_idx == 0:
                pad = torch.zeros((num_mics, 256), device=noisy.device)

                chunk_start_position = chunk_idx * chunk_hop_length
                chunk_end_position = chunk_start_position + chunk_length

                # concat([(8, 256), (..., ... + chunk_length)])
                noisy_chunk = torch.cat((pad, noisy[:, chunk_start_position:chunk_end_position]), dim=1)
                enhanced_chunk = self.model(noisy_chunk.unsqueeze(0))
                enhanced_chunk = torch.squeeze(enhanced_chunk)
                enhanced_chunk = enhanced_chunk[256:]

                # Save the prior half chunk,
                cur = enhanced_chunk[:chunk_length // 2]

                # only for the 1st chunk,no overlap for the very 1st chunk prior half
                prev = enhanced_chunk[chunk_length // 2:] * win[chunk_length // 2:]
            else:
                # use the previous noisy data as the pad
                pad = noisy[:, (chunk_idx * chunk_hop_length - 256):(chunk_idx * chunk_hop_length)]

                chunk_start_position = chunk_idx * chunk_hop_length
                chunk_end_position = chunk_start_position + chunk_length

                noisy_chunk = torch.cat((pad, noisy[:8, chunk_start_position:chunk_end_position]), dim=1)
                enhanced_chunk = self.model(noisy_chunk.unsqueeze(0))
                enhanced_chunk = torch.squeeze(enhanced_chunk)
                enhanced_chunk = enhanced_chunk[256:]

                # 使用这个窗函数来对拼接的位置进行平滑？
                enhanced_chunk = enhanced_chunk * win[:len(enhanced_chunk)]

                tmp = enhanced_chunk[:chunk_length // 2]
                cur = tmp[:min(len(tmp), len(prev))] + prev[:min(len(tmp), len(prev))]
                prev = enhanced_chunk[chunk_length // 2:]

            if enhanced is None:
                enhanced = cur
            else:
                enhanced = torch.cat((enhanced, cur), dim=0)

        enhanced = enhanced[:noisy.shape[1]]
        return enhanced.detach().squeeze(0).cpu().numpy()

    @torch.no_grad()
    def inference_once(self, waveform):
        inference_type = self.inference_config["type"]
        inference_args = self.inference_config["args"]

        waveform = torch.tensor([waveform])
        enhanced, model_rtf = getattr(self, inference_type)(waveform.to(self.device), inference_args)

        if abs(enhanced).any() > 1:
            print(f"Warning: current enhanced sound is not in the range [-1, 1]")

        amp = np.iinfo(np.int16).max
        enhanced = np.int16(0.8 * amp * enhanced / np.max(np.abs(enhanced)))

        return enhanced

if __name__ == '__main__':
    a = torch.rand(10, 2, 161, 200)
    print(cumulative_norm(a).shape)
