
# Adaptive-FSN

This Git repository is for the official PyTorch implementation of **"[Adaptive-FSN: Integrating Full-Band Extraction and Adaptive Sub-Band Encoding for Monaural Speech Enhancement](https://ieeexplore.ieee.org/document/10023439/)"**,  accepted by SLT 2022.

📜[[Full Paper](https://ieeexplore.ieee.org/document/10023439)] ▶[[Demo](#)] 💿[[Checkpoint](https://drive.google.com/file/d/1bnfapAm0OX6fiyDDL53OvwvKL_J8_yvk/view)]



## Requirements

\- Linux or macOS 

\- python>=3.8

\- Anaconda or Miniconda

\- NVIDIA GPU + CUDA CuDNN (CPU can also be supported)



### Environment && Installation

Install Anaconda or Miniconda, and then install conda and pip packages:

```shell
# Create conda environment
conda create --name speech_enhance python=3.8
conda activate speech_enhance

# Install conda packages
# For CUDA 10.2
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
# OR CUDA 11.3
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install pip packages
pip install -r requirements.txt

# (Optional) If you want to load "mp3" format audio in your dataset
conda install -c conda-forge ffmpeg
```



### Quick Usage

Clone the repository:

```shell
git clone https://github.com/samx81/Adaptive-FSN.git
cd Adaptive-FSN
```

Download the [pre-trained checkpoint](https://drive.google.com/file/d/1bnfapAm0OX6fiyDDL53OvwvKL_J8_yvk/view), and input commands:

```shell
source activate speech_enhance
python -m speech_enhance.tools.inference \
  -C pretrained/inference_adaptive-fsn.toml \
  -M $MODEL_PTH \
  -I $INPUT_DIR \
  -O $OUTPUT_DIR
```

<br/> 

## Start Up

### Clone

```shell
git clone https://github.com/samx81/Adaptive-FSN.git
cd Adaptive-FSN
```

### Data preparation

#### VCTK Training data

Please prepare your VCTK data in [scp format](https://kaldi-asr.org/doc/data_prep.html) like:

`<recording-id> <extended-filename>`

Then specify the path in `config/train_vctk.toml`.

#### DNS Challenge Training & Testing Data

Please follow [FullSubNet-Plus Repo](https://github.com/hit-thusz-RookieCJ/FullSubNet-plus) to generate DNS Data.


### Training

First, you need to modify the various configurations in `config/train.toml` for training.

Then you can run training:

```shell
source activate speech_enhance
python -m speech_enhance.tools.train \
        -C <config_path> \
        -N <n_gpu=1> [-R (resume training)]
```



### Inference

After training, you can enhance noisy speech.  Before inference, you first need to modify the configuration in `config/inference.toml`.

You can also run inference:

```shell
source activate speech_enhance
python speech_enhance\tools\inference.py \
      -C config/inference.toml \
      -M logs/<tag>/checkpoint/<best_model|latest_model>.tar
```


### Eval

Calculating bjective metrics (SI_SDR, STOI, WB_PESQ, NB_PESQ, etc.) :

```shell
bash custom_metrics.sh <exp_tag> <ref_dir> <enh_dir>
```

Obtain subjective scores (DNS_MOS):
```shell
python ./speech_enhance/tools/dns_mos.py --testset_dir $YOUR_TESTSET_DIR --score_file $YOUR_SAVE_DIR
```