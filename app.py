import streamlit as st
import os, time
import librosa, librosa.display
from matplotlib import pyplot as plt

import sys
from pathlib import Path

import numpy as np
import toml
import gdown

sys.path.append(os.path.abspath(os.path.join("speech_enhance")))
from audio_zen.utils import initialize_module

cfg_path = "pretrained/inference_adaptive-fsn.toml"
ckpt_url = "https://drive.google.com/u/2/uc?id=1bnfapAm0OX6fiyDDL53OvwvKL_J8_yvk"
ckpt_path = "pretrained/Adaptive-FSN_VCTK.tar"

def plot_wave(y, sr):
    fig, ax = plt.subplots()
    ax.set_box_aspect(0.3)

    img = librosa.display.waveshow(y, sr=sr, ax=ax)
    fig.tight_layout()

    return plt.gcf()

@st.cache_resource(show_spinner="Loading Model...")
def load_inferencer(cfg_path, ckpt_path, ckpt_url):
    config = toml.load(cfg_path)

    if not Path(ckpt_path).exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            gdown.download(ckpt_url, ckpt_path)
        
    inferencer_class = initialize_module(config["inferencer"]["path"], initialize=False)
    inferencer = inferencer_class(config, ckpt_path, "",file_mode=True)

    return inferencer

st.title('Adaptive-FSN Demo')
inferencer = load_inferencer(cfg_path, ckpt_path, ckpt_url)

model_type = st.selectbox("Model Selection",
                ['Adaptive-FSN (VCTK)',
                 'Adaptive-FSN (Magnitude) (DNSChallenge)',
                 'FullSubNet-Plus (VCTK)'])

if st.button("Load"):
    if model_type == "Adaptive-FSN (VCTK)":
        desire_model_path = "pretrained/Adaptive-FSN_VCTK.tar"
        desire_cfg_path   = "pretrained/inference_adaptive-fsn.toml"
    if model_type == 'Adaptive-FSN (Magnitude) (DNSChallenge)':
        desire_model_path = "pretrained/Adaptive-Magnitude-FSN_DNS.tar"
        desire_cfg_path   = "pretrained/inference_adaptive_mag.toml"
        desire_model_url  = "https://drive.google.com/u/2/uc?id=1aMLP6CBNEM2BGCDburt1DxSYqHz4heig"
    if model_type == 'FullSubNet-Plus (VCTK)':
        desire_model_path = "pretrained/FullSubNet-Plus_VCTK.tar"
        desire_cfg_path   = "pretrained/inference_fsn.toml"
        desire_model_url  = "https://drive.google.com/u/2/uc?id=1o2bn2aJ5aYG4pMKsjQo8SigFABFDqdvF"
    if not Path(desire_model_path).exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            gdown.download(desire_model_url, desire_model_path)
    inferencer = load_inferencer(
        desire_cfg_path, desire_model_path, ckpt_url)


_, container, _ = st.columns([1,4,1])

with container:
    f = st.file_uploader("Upload audio to denoise", type=[".wav",".mp3"])
    btn = False
    sr = 16000
    if f is not None:
        _, f_ext = os.path.splitext(f.name)

        if f_ext == '.wav': fmt = "wav"
        elif f_ext == 'mp3': fmt = "mp3"

        y, sr = librosa.load(f, sr=None)
        st.audio(y, format=fmt, sample_rate=sr)
        with st.spinner("Generating wavegram..."):
            st.pyplot(plot_wave(y, sr))

        btn = st.button("Enhance!")
    if btn:
        with st.spinner('Wait for it...'):
            y = y.astype(np.float32)
            enh = inferencer.inference_once(y)
            enh = enh.astype(np.float32)
        st.write('Enhanced audio')
        st.audio(enh, sample_rate=sr)
        st.pyplot(plot_wave(enh, sr))
        
