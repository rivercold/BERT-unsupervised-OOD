#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

from multiprocessing import cpu_count

import torch
import numpy as np

from simpletransformers.classification.classification_utils import (
    InputExample,
    convert_examples_to_features,
)
from torch.utils.data import DataLoader, SequentialSampler
from simpletransformers.classification import ClassificationModel

from tqdm import tqdm
import pdb

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer
)
from simpletransformers.classification.transformer_models.bert_model import BertForSequenceClassification
from simpletransformers.classification.transformer_models.roberta_model import RobertaForSequenceClassification
from simpletransformers.config.global_args import global_args

class our_model(ClassificationModel):
    def __init__(self, model_type, model_name, num_labels=None, weight=None, args=None, use_cuda=True, cuda_device=-1, **kwargs):
        # super().__init__(model_type, model_name, num_labels, weight, args, use_cuda, cuda_device, **kwargs)
        MODEL_CLASSES = {
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
            "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
        }

        if args and 'manual_seed' in args:
            random.seed(args['manual_seed'])
            np.random.seed(args['manual_seed'])
            torch.manual_seed(args['manual_seed'])
            if 'n_gpu' in args and args['n_gpu'] > 0:
                torch.cuda.manual_seed_all(args['manual_seed'])

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if num_labels:
            self.config = config_class.from_pretrained(model_name, num_labels=num_labels, output_hidden_states=True, **kwargs)
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name, output_hidden_states=True, **kwargs)
            self.num_labels = self.config.num_labels
        self.weight = weight

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if self.weight:

            self.model = model_class.from_pretrained(
                model_name, config=self.config, weight=torch.Tensor(self.weight).to(self.device), **kwargs,
            )
        else:
            self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)

        self.results = {}

        self.args = {
            "sliding_window": False,
            "tie_value": 1,
            "stride": 0.8,
            "regression": False,
        }

        self.args.update(global_args)

        if not use_cuda:
            self.args["fp16"] = False

        if args:
            self.args.update(args)

        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=self.args["do_lower_case"], **kwargs)

        self.args["model_name"] = model_name
        self.args["model_type"] = model_type

        if self.args["wandb_project"] and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args["wandb_project"] = None
        print ("Use SimpleTransformers ClassificationModel")
    

    def sample_X_estimator(self, input_sentences, use_cls=True):
        device = self.device 
        model = self.model 
        args = self.args 

        self._move_model_to_device()
        eval_examples = [InputExample(i, text, None, 0) for i, text in enumerate(input_sentences)]
        eval_dataset = self.load_and_cache_examples(
                eval_examples, evaluate=True, multi_label=False, no_cache=True
            )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=32)
        
        import sklearn.covariance
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        
        model.eval()
        all_layer_features = []
        num_layers = 13
        for i in range(num_layers):
            all_layer_features.append([])
        
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = self._get_inputs_dict(batch)
            with torch.no_grad():
                batch_all_features = self.get_hidden_features(**inputs, use_cls=use_cls)
                for i in range(num_layers):
                    all_layer_features[i].append(batch_all_features[i].cpu())  # save gpu memory
        
        mean_list = []
        precision_list = []
        for i in range(num_layers):
            all_layer_features[i] = torch.cat(all_layer_features[i], axis=0)
            sample_mean = torch.mean(all_layer_features[i], axis=0)
            X = all_layer_features[i] - sample_mean
            group_lasso.fit(X.numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float()
            mean_list.append(sample_mean.to(device))
            precision_list.append(temp_precision.to(device))

        return mean_list, precision_list


    def get_unsup_Mah_score(self, test_sentences, sample_mean, precision, use_cls=True):
        device = self.device 
        model = self.model 
        args = self.args 

        self._move_model_to_device()
        test_examples = [InputExample(i, text, None, 0) for i, text in enumerate(test_sentences)]
        test_dataset = self.load_and_cache_examples(
                test_examples, evaluate=True, multi_label=False, no_cache=True)

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=128)

        model.eval()
        num_layers = 13
        total_mah_scores = []
        for i in range(num_layers):
            total_mah_scores.append([])

        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = self._get_inputs_dict(batch)
            with torch.no_grad():
                batch_all_features = self.get_hidden_features(**inputs, use_cls=use_cls)
            
            for i in range(len(batch_all_features)):
                batch_sample_mean = sample_mean[i]
                out_features = batch_all_features[i]
                zero_f = out_features - batch_sample_mean
                gaussian_score = -0.5 * ((zero_f @ precision[i]) @ zero_f.t()).diag()
                total_mah_scores[i].extend(gaussian_score.cpu().numpy())

        for i in range(len(total_mah_scores)):
            total_mah_scores[i] = np.expand_dims(np.array(total_mah_scores[i]), axis=1)
        return np.concatenate(total_mah_scores, axis=1)


    def get_alternative_distance_score(self, test_sentences, sample_mean, use_cls=True):
        device = self.device 
        model = self.model 
        args = self.args 

        self._move_model_to_device()
        test_examples = [InputExample(i, text, None, 0) for i, text in enumerate(test_sentences)]
        test_dataset = self.load_and_cache_examples(
                test_examples, evaluate=True, multi_label=False, no_cache=True)

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=128)
        # cosine_sim_nn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        model.eval()
        num_layers = 13
        total_scores = []
        for i in range(num_layers):
            total_scores.append([])

        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = self._get_inputs_dict(batch)
            with torch.no_grad():
                # batch_all_features = model.feature_list(**inputs, use_cls=use_cls)
                batch_all_features = self.get_hidden_features(**inputs, use_cls=use_cls)
            
            for i in range(len(batch_all_features)):
                batch_sample_mean = sample_mean[i]
                out_features = batch_all_features[i]
                zero_f = out_features - batch_sample_mean    # bs x hidden_dim
                l2_distance = torch.norm(zero_f, dim=1)
                total_scores[i].extend(l2_distance.cpu().numpy())
                # cosine_sim = cosine_sim_nn(batch_sample_mean.unsqueeze(0), out_features)
                # total_mah_scores[i].extend(cosine_sim.cpu().numpy())

        for i in range(len(total_scores)):
            total_scores[i] = np.expand_dims(np.array(total_scores[i]), axis=1)
        
        return np.concatenate(total_scores, axis=1)


    def get_one_layer_feature(self, input_sentences, use_layer=-1, use_cls=True):
        device = self.device 
        model = self.model
        args = self.args 

        self._move_model_to_device()
        eval_examples = [InputExample(i, text, None, 0) for i, text in enumerate(input_sentences)]
        eval_dataset = self.load_and_cache_examples(
                eval_examples, evaluate=True, multi_label=False, no_cache=True
            )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])
        
        latent_features = []
        model.eval()
        for batch in tqdm(eval_dataloader, disable=args["silent"]):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)
                outputs = self.get_hidden_features(**inputs, use_cls=use_cls)
                latent_features.append(outputs[use_layer].detach())
        return torch.cat(latent_features, dim=0).data.cpu().numpy()


    def get_hidden_features(self, input_ids=None,  attention_mask=None, token_type_ids=None, labels=None,
        position_ids=None, head_mask=None, use_cls=True):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        all_hidden_feats = outputs[1]   # list (13) of bs x length x hidden
        all_feature_list = []
        for i in range(len(all_hidden_feats)):
            if use_cls:
                if "roberta" in str(type(self.model)):   # a dumb way... to use the pooler layer
                    pooled_feats = self.model.roberta.pooler(all_hidden_feats[i]).detach()  # bs x max_len x 768 -> bs x 768
                else:
                    pooled_feats = self.model.bert.pooler(all_hidden_feats[i]).detach()  # bs x max_len x 768 -> bs x 768
                # pooled_feats = all_hidden_feats[i][:,0].detach().data.cpu()  # bs x max_len x 768 -> bs x 768
                # print (pooled_feats.shape)
            else:
                pooled_feats = torch.mean(all_hidden_feats[i], dim=1, keepdim=False).detach()  # bs x max_len x 768 -> bs x 768
            all_feature_list.append(pooled_feats.data)   # 13 list of bs x 768
        return all_feature_list