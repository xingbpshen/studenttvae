import comet_ml
import argparse
import os
import yaml
import traceback
import sys
import torch
import numpy as np
from runners.run_vae import VAERunner


# Code from DDIM by Song et al. (https://github.com/ermongroup/ddim)
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# Code adapted from DDIM by Song et al. (https://github.com/ermongroup/ddim)
def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data, e.g., ./log."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder and the comet trial name.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--vae",
        type=str,
        required=True,
        help="VAE: gaussian | student-t",
    )
    parser.add_argument("--train", action="store_true", help="Whether to train the model")
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument("--d_shift", action="store_true", help="Whether to test on the shifted data")
    parser.add_argument("--sample", action="store_true", help="Whether to sample from the model")
    parser.add_argument(
        "--comet", action="store_true", help="Whether to use comet.ml"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use: cuda:idx | cpu"
    )

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, new_config


# Code adapted from DDIM by Song et al. (https://github.com/ermongroup/ddim)
def main():
    args, config = parse_args_and_config()

    try:
        runner = VAERunner(args, config)
        if args.train:
            if os.path.exists(runner.log_path):
                # ask if user wants to overwrite the existing folder
                response = input(f"Folder {runner.log_path} already exists. Overwrite? (y/n): ")
                # case insensitive check
                response = response.lower()
                if response != "y":
                    return 0
                elif response == "y":
                    # remove the existing folder and create a new one
                    os.system(f"rm -r {runner.log_path}")
                    os.makedirs(runner.log_path)
                    runner.train()
            else:
                os.makedirs(runner.log_path)
                runner.train()
        elif args.test:
            runner.test(d_shift=args.d_shift)
        elif args.sample:
            runner.sample()
        else:
            raise ValueError("Invalid mode")
    except Exception:
        print(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
