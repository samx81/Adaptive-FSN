import argparse
import os, sys
from pathlib import Path

import numpy as np
import librosa
import toml

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from audio_zen.utils import initialize_module


def main(config, checkpoint_path, output_dir):
    inferencer_class = initialize_module(config["inferencer"]["path"], initialize=False)
    inferencer = inferencer_class(
        config,
        checkpoint_path,
        output_dir
    )
    inferencer()

def main_filemode(config, checkpoint_path, input_dir, output_dir):
    inferencer_class = initialize_module(config["inferencer"]["path"], initialize=False)
    inferencer = inferencer_class(
        config,
        checkpoint_path,
        output_dir,
        file_mode=True
    )
    noisy_file_path_list = librosa.util.find_files(Path(input_dir).expanduser().absolute().as_posix())
    for noisy_file_path in noisy_file_path_list:
        noisy_y = librosa.load(noisy_file_path, sr=inferencer.sr)[0]
        noisy_y = noisy_y.astype(np.float32)
        enhanced = inferencer.inference_once(noisy_y)

        print(enhanced.shape)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("-C", "--configuration", type=str, required=True, help="Config file.")
    parser.add_argument("-M", "--model_checkpoint_path", type=str, required=True, help="The path of the model's checkpoint.")
    parser.add_argument('-I', '--dataset_dir_list', help='delimited list input',
                        type=lambda s: [item.strip() for item in s.split(',')])
    parser.add_argument("-O", "--output_dir", type=str, required=True, help="The path for saving enhanced speeches.")
    args = parser.parse_args()

    configuration = toml.load(args.configuration)
    checkpoint_path = args.model_checkpoint_path
    output_dir = args.output_dir
    if len(args.dataset_dir_list) > 0:
        print(f"use specified dataset_dir_list: {args.dataset_dir_list}, instead of in config")
        configuration["dataset"]["args"]["dataset_dir_list"] = args.dataset_dir_list

    # main(configuration, checkpoint_path, output_dir)
    main_filemode(configuration, checkpoint_path, args.dataset_dir_list[0], output_dir)
