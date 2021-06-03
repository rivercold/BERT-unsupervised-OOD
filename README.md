# BERT-unsupervised-OOD
Code for ACL 2021 paper "Unsupervised Out-of-Domain Detection via Pre-trained Transformers" by Keyang Xu, Tongzheng Ren, Shikun Zhang, Yihao Feng and Caiming Xiong.

## Requirements

* Python 3.7.3
* PyTorch 1.2
* Transformers 2.7.0
* simpletransformers 0.22.1

I do notice this repo is not compatabile with the newest version of Transformers (4.6.1) and simpletransformers (0.61.6). I will try to address this issue in a new branch ASAP. 

## Overview
![An overview of using Mahalanobis distance features (MDF) extracted from a pre-trained transformer $f$ to detect out-of-domain data.](overview.jpg)

## To run the models
Use the command 
```
python ood_main.py \
  --method MDF \
  --data_type clinic \
  --model_class bert
```

To run baselines, change MDF to one from ``single_layer_bert``, ``tf-idf`` and ``MSP`` (should load a BCAD mdoel). 


## Fine-tuning BERT with BCAD and IMLM

### In-domain Masked Language Model (IMLM)
```
python finetune_bert.py \
    --type finetune_imlm \ 
    --data_type clinic \ 
    --model_class bert
```

### Binary   classification   with   auxiliary   dataset (BCAD)
```
python finetune_bert.py \
    --type finetune_binary \ 
    --data_type clinic \ 
    --model_class bert
    --load_path xxx
```
The ``load_path`` can be the output of IMLM fine-tuning. If no ``load_path`` is specified, then pre-trained bert model is used. 

You can use the fine-tuned model for OOD detection by adding the ``load_path`` parameter, e.g., 

```
python ood_main.py \
  --method MDF \
  --data_type clinic \
  --model_class roberta \
  --load_path  ./models/roberta_clinic_ft_IMLM_BCAD
```

You can also downloaded our fine-tuned RoBERTa (IMLM+BCAD) models for SST and CLINIC150 [here](https://drive.google.com/drive/folders/1CVKEITegBMaPRwfIUtNktBhpTAjzqhYW?usp=sharing). 




## Citations
<!-- 
```
@inproceedings{xu2021unsupervised,
  title={Leveraging Just a Few Keywords for Fine-Grained Aspect Detection Through Weakly Supervised Co-Training},
  author={Karamanolakis, Giannis and Hsu, Daniel and Gravano, Luis},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={4603--4613},
  year={2019}
}
``` -->