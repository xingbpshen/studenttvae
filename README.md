<h1 align="center">
Scalability of Student-t VAEs for Robust Density Estimation
</h1>

This repository contains the code for the report "On Scalability of Student-t Variational Autoencoders for Robust Density Estimation: A Case Study on Images" by X. Shen. This report is a part of the course project for the ECSE 626 Statistical Computer Vision at McGill University.

Please note that all results in the report are fully reproducible using the code in this repository.

## Acknowledgements
The code is based on the paper "[Student-t Variational Autoencoder for Robust Density Estimation](https://www.ijcai.org/proceedings/2018/0374.pdf)" by Takahashi et al. published at IJCAI 2018. In addition, the folder hierarchies and a very small portion of the code for arguments and configurations parsing is adapted from the [DDIM code base](https://github.com/ermongroup/ddim), we have properly cited the original authors in the code. We sincerely thank the authors for making their code available.

## Requirements
The requirements are listed in `requirements.txt`. You can install them using the following command:
```
pip install -r requirements.txt
```

## Datasets
If you are the grader of the course ECSE 626, please directly contact the author (me) to obtain the dataset for fully reproducing the results in the report. Otherwise, the datasets can be downloaded from the following links, in this case, different data splits may be used, and the results may vary slightly:

- Statlog Landsat Satellite ([link](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite))
- CelebA, aligned and cropped images ([link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))
- BIMCV COVID-19 ([link](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/))

## Usage
### Modifying the configurations
The configurations for the experiments are stored in the `configs` folder. You can modify the configurations to run different experiments. In a minimum case, you need to modify the `data.data_dir` in the configuration files to point to the correct dataset path.

### Reproducing the results
For a specific dataset, you can run the following command to reproduce the results in the report.

1. First, train the model using the following command:
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {TAG} --vae {VAE_TYPE} --train
```
where `DATASET` is the dataset name (e.g., `statlog` or `celeba_gender` or `bimcv`), `PROJECT_PATH` is the path to store the experiment results (e.g., `./log`), `TAG` is the tag for the experiment (you can name any as you wish but you need to remember this for the testing command), and `VAE_TYPE` is the type of VAE to use (either `gaussian` or `student-t`). The `--train` option is used to train the model.

2. Then, test the model using the following command:
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {TAG} --vae {VAE_TYPE} --test
```
where `DATASET`, `PROJECT_PATH`, `TAG`, and `VAE_TYPE` are the same as above. The `--test` option is used to test the model. To reproduce the results in Table.3 (Male faces and COVID-19 patients) of the report, you can use the following command:
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {TAG} --vae {VAE_TYPE} --test --d_shift
```

### Additional options
You can use the following additional options to control the training and testing process:
- `--comet` to log the experiment results to Comet.ml. You need to set environment variables `COMET_API_KEY` and `COMET_WORKSPACE` to use this option, a default project name "scalability-student-t-vae" will be used.
- `--device {DEVICE}` to specify the device to use (e.g., `{DEVICE}` can be `cuda:0` or `cpu`), currently only supports single GPU training and testing.

## References
- Hiroshi Takahashi, Tomoharu Iwata, Yuki Yamanaka, Masanori Yamada, and Satoshi Yagi. Student-t variational autoencoder for robust density estimation. In _Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence_, pages 2696–2702, 2018.