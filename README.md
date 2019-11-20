# Temporal Binding Network


This repository implements the model proposed in the paper:

Evangelos Kazakos, Arsha Nagrani, Andrew Zisserman, Dima Damen, <strong>EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition</strong>, <em>ICCV</em>, 2019

[Project's webpage](https://ekazakos.github.io/TBN/)

[ArXiv paper](https://arxiv.org/abs/1908.08498)

## Citing

When using this code, kindly reference:

```
@InProceedings{kazakos2019TBN,
    author    = {Kazakos, Evangelos and Nagrani, Arsha and Zisserman, Andrew and Damen, Dima},
    title     = {EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition},
    booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2019}
}
```

## Requirements

* Install project's requirements in a separate conda environment. In your terminal: `$ conda env create -f environment.yml`. 
* CUDA 10.0

## Data preparation

### Visual data

This step assumes that you've downloaded the RGB and Flow frames of EPIC dataset using [this script](https://github.com/epic-kitchens/download-scripts/blob/master/download_frames_rgb_flow.sh), and your copy of the dataset has the same folder structure 
provided in the script (which can be found [here](https://data.bris.ac.uk/data/dataset/3h91syskeag572hl6tvuovwv4d)). Also you should untar each video's frames in its corresponding folder. 

`dataset.py` uses a unified folder structure for all datasets, which is the same as the one used in the [TSN code](https://github.com/yjxiong/tsn-pytorch). Example of the folder structure for RGB and Flow:

```
├── dataset_root
|   ├── video1
|   |   ├── img_0000000000
|   |   ├── x_0000000000
|   |   ├── y_0000000000
|   |   ├── .
|   |   ├── .
|   |   ├── .
|   |   ├── img_0000000100
|   |   ├── x_0000000100
|   |   ├── y_0000000100
|   ├── .
|   ├── .
|   ├── .
|   ├── video10000
|   |   ├── img_0000000000
|   |   ├── x_0000000000
|   |   ├── y_0000000000
|   |   ├── .
|   |   ├── .
|   |   ├── .
|   |   ├── img_0000000250
|   |   ├── x_0000000250
|   |   ├── y_0000000250
```
        
To map the folder structure of EPIC to the above folder structure I've used symlinks. Use the following script to convert
the original folder structure of EPIC to the folder structure above:

```
python preprocessing_epic/symlinks.py /path/to/rgb_flow/ /path/to/output
```

### Audio data

This step assumes that you've downloaded the videos of EPIC using [this script](https://github.com/epic-kitchens/download-scripts/blob/master/download_videos.sh).

To extract the audio from the videos, run:

```
python preprocessing_epic/extract_audio.py /path/to/videos /path/to/ouput
```

To load the audio in `dataset.py`, Im using a dictionary, where the keys are the video names and the values are the extracted audio from the previous step. To save the extracted audio into a dictionary, run:

```
python preprocessing_epic/wav_to_dict.py /path/to/audio /path/to/output
```

If you don't want to save the audio in a dictionary, and prefer to load the wav files directly in `dataset.py`, you can
change set `use_audio_dict=False` in `TBNDataset` in `dataset.py`.

### Pretrained models

To download the pretrained models, run the following on your terminal

```
$ cd pretrained
$ bash download.sh
```

Two files will be downloaded:

* **epic_tbn_rgbflowaudio.pth.tar**, which is the full TBN model(RGB, Flow, Audio) trained on EPIC-Kitchens, which we use to report results in our paper 
* **kinetics_tsn_flow.pth.tar**, which is a TSN Flow model, trained on Kinetics, downloaded from [here](http://yjxiong.me/others/kinetics_action/). The original model was on Caffe and I converted it to a PyTorch model. This can be used for initialising the Flow stream from Kinetics when training TBN, as we observed an increase in performance in preliminary experiments in comparison to initialising Flow from ImageNet.

## Train/evaluate with other datasets

Basic steps:

1. Extract the audio in a similar way to the one that I've shown above (.wav files for all dataset in a single folder). Have a look at preprocessing_epic/extract_audio.py for help.
2. Visual data should have the same folder structure as the one that I've shown above. To do that, map your original folder structure to the one above using symlinks, similarly to epic_preprocessing/symlinks.py
3. In both train.py and test.py, register the number of classes of your dataset in the variable num_class at the top of main().
1. Under video_records/ create *your_record.py* which should inherit from VideoRecord. This should parse the lines of a file that contains info about your dataset (paths, labels etc). Have a look at epic_record.py as an example.
2. Add your dataset in \_parse_list() in dataset.py, by parsing each line of list_file ans storing it to a list, where list_file is the file that contain info for your dataset.


## Training

To train the full RGB, Flow, Audio model, run:
```
python train.py epic RGB Flow Spec --train_list train_val/EPIC_train_action_labels.pkl --val_list train_val/EPIC_val_action_labels.pkl --visual_path /path/to/rgb+flow --audio_path /path/to/audio --arch BNInception 
--num_segments 3 --dropout 0.5 --epochs 80 -b 128 --lr 0.01 --lr_steps 60 --gd 20 --partialbn --eval-freq 1 -j 40 
--pretrained_flow_weights
```

**In the paper, results are reported by training on the whole training set. The pretrained model in `pretrained/` is the result of training in the whole training set** Train/val sets where used for development and hyperparam tuning. To train on the whole dataset, run:
```
python train.py epic RGB Flow Spec --visual_path /path/to/rgb+flow --audio_path /path/to/audio --arch BNInception 
--num_segments 3 --dropout 0.5 --epochs 80 -b 128 --lr 0.01 --lr_steps 60 --gd 20 --partialbn --eval-freq 1 -j 40 
--pretrained_flow_weights
```

When `--train_list` is not used, the original training set is loaded (without any splitting). When using the whole training set, validation set cannot be used. 

Individual modalities can be trained, as well as any combination of 2 modalities. 
To train audio, run:
```
python train.py epic Spec --train_list train_val/EPIC_train_action_labels.pkl --val_list train_val/EPIC_val_action_labels.pkl 
--audio_path /path/to/audio --arch BNInception --num_segments 3 
--dropout 0.5 --epochs 80 -b 128 --lr 0.001 --lr_steps 60 --gd 20 --partialbn --eval-freq 1 -j 40 
```

To train RGB, run:
```
python train.py epic RGB  --train_list train_val/EPIC_train_action_labels.pkl --val_list train_val/EPIC_val_action_labels.pkl 
--visual_path /path/to/rgb+flow --arch BNInception --num_segments 3 --dropout 0.5 
--epochs 80 -b 128 --lr 0.01 --lr_steps 60 --gd 20 --partialbn --eval-freq 1 -j 40 
```

To train flow, run:
```
python train.py epic Flow  --train_list train_val/EPIC_train_action_labels.pkl --val_list train_val/EPIC_val_action_labels.pkl 
--visual_path /path/to/rgb+flow --arch BNInception --num_segments 3 --dropout 0.5 
--epochs 80 -b 128 --lr 0.001 --lr_steps 60 --gd 20 --partialbn --eval-freq 1 -j 40 --pretrained_flow_weights
```

Example of training RGB+Audio (any other combination can be used):
```
python train.py epic RGB Spec --train_list train_val/EPIC_train_action_labels.pkl --val_list train_val/EPIC_val_action_labels.pkl --visual_path /path/to/rgb+flow --audio_path /path/to/audio --arch BNInception 
--num_segments 3 --dropout 0.5 --epochs 80 -b 128 --lr 0.01 --lr_steps 60 --gd 20 --partialbn --eval-freq 1 -j 40 
```

`EPIC_train_action_labels.pkl` and `EPIC_val_action_labels.pkl` can be found under `train_val/`. They are the result of spliting the original [EPIC_train_action_labels.pkl](https://github.com/epic-kitchens/annotations/blob/master/EPIC_train_action_labels.csv) into a training and a validation set, by randomly holding out  one untrimmed video from each participant for the 14 kitchens (out of 32) with the largest number of untrimmed videos.  

## Testing

To compute scores, save scores and labels, and print the accuracy of the validation set using the full modalities, run:

```
python test.py epic RGB Flow Spec path/to/checkpoint --test_list train_val/EPIC_val_action_labels.pkl --visual_path /path/to/rgb+flow --audio_path /path/to/audio --arch BNInception --scores_root scores/ --test_segments 25 --test_crops 1  --dropout 0.5 -j 40
```

To compute and save scores of the test sets (S1/S2) (since we do not have access to the labels), run:

```
python test.py epic RGB Flow Spec path/to/checkpoint --visual_path /path/to/rgb+flow --audio_path /path/to/audio --arch BNInception --scores_root scores/ --test_segments 25 --test_crops 1  --dropout 0.5 -j 40
```

When `--test_list` is not provided, the timestamps of S1/S2 are automatically loaded. 

Similarly testing can be done for any combination of modalities, or individual modalities.

Furthermore, you can use `fuse_results_epic.py` to fuse modalities' scores with late fusion, assuming that you trained individual modalities (similarly to TSN). Lastly, `submission_json.py` can be used for preparing your scores in json format to submit them in the EPIC-Kitchens Action Recognition Challenge. 

## Validation set results

The following table contains the results of training and evaluating on the splits from `train_val/`.

**Top-1 Accuracy**:

| VERB | NOUN | ACTION
| ---- | ---- | ------
| 63.31 | 46.00 | 34.83 |

**Top-5 Accuracy**:

| VERB | NOUN | ACTION
| ---- | ---- | ------
| 88.29 | 68.31 | 54.09 |
 
## License 

The code is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).
