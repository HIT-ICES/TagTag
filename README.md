
# README

## Introduction
This repo is the source code for paper “TagTag: A Novel Framework for Service Tags Recommendation and Service Missing Tag Prediction”

The code is oranized as follows: 
```
root
├─Ablation_Experiment		TagTag with different layers of GCN, TagTag only for recommendation task. TagTag only for prediction task and αTagTag
│          
├─Baseline_Recommendation	text-only methods trained only for recommendation task
│      
├─Baseline_Joint			text-only methods trained for two task
│      
└─Baseline_Prediction		text-only methods trained only for prediction task
|
└─Data						dataset for experiment
|
└─TagTag					TagTag method
```

## Usage
### Requirements
The following packages are required:

```
torch==1.10.0
pytorch_lightning==1.5.7
numpy==1.19.5
gensim==3.8.3
nltk==3.6.5
scikit_learn==1.0.2
transformers==4.15.0
torchmetrics==0.6.2
torch_geometric==2.0.4

```

### Train models
- Clone this project.
- Go into the root of repo and install the required package listed in `requirements.txt` by:
```commandline
pip install -r requirement.txt
```
- Use `python` command to train and test the model. For example:
```commandline
python TagTag/TagTag.py
```

