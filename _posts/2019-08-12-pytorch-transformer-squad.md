---
layout: post
title:  "Machine Comprehension with pytorch-transformers"
date:   2019-08-12 12:52:41 -0300
image:
  path: /images/transformer.jpg
categories: machine_learning nlp pytorch
---

# Step by step finetuning and configuration question and answering with pytorch-transformers

I being using question and answering system at work and also for some personal POCs, and I'm really impressed how these algorithms evolved recently. My first interaction with QA algorithms was with the BIDAF model (Bidirectional Attention Flow) [1](#Bidaf) from the great [AllenNLP](https://allennlp.org/) team. It was back in 2017, and [ELMo](https://allennlp.org/elmo) was not even used in this BIDAF model (I believe they were using GLove vectors in this first model). Since then, a lot of stuff is happen in NLP arena, such as ELMo, the [Transformer](https://arxiv.org/abs/1706.03762), [BERT](https://arxiv.org/abs/1810.04805),  what is this happening


$$
Attention(Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}})V$
$

## Finetuning XLNet

```
 python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --config_name medium_bert.json \
  --evaluate_during_training \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --save_steps 10000 \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /Users/rsilvei/repos/adp-e-chat-sage-mc-xlnet/finetune/finetuned_bert \
  --overwrite_output_dir \
  --overwrite_cache
```

## Finetuning BERT
```
 python run_squad.py \
  --model_type xlnet \
  --model_name_or_path xlnet-large-cased \
  --do_train \
  --do_eval \
  --config_name xlnet_m2.json \
  --evaluate_during_training \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --save_steps 10000 \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /home/roberto/tmp/finetuned_xlnet \
  --overwrite_output_dir \
  --overwrite_cache
```

Config `xlnet_medium.json`

```json
{
  "attn_type": "bi",
  "bi_data": false,
  "clamp_len": -1,
  "d_head": 64,
  "d_inner": 3072,
  "d_model": 768,
  "dropout": 0.1,
  "end_n_top": 5,
  "ff_activation": "gelu",
  "finetuning_task": null,
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-12,
  "mem_len": null,
  "n_head": 12,
  "n_layer": 16,
  "n_token": 32000,
  "num_labels": 2,
  "output_attentions": false,
  "output_hidden_states": false,
  "reuse_len": null,
  "same_length": false,
  "start_n_top": 5,
  "summary_activation": "tanh",
  "summary_last_dropout": 0.1,
  "summary_type": "last",
  "summary_use_proj": true,
  "torchscript": false,
  "untie_r": true
}
```

## Results on SQuAD 1.1
total 5 epochs

| Model                                                | Config File              | EM       | F1      | Epochs  |
| -------------                                        |------------              |--------- |-------- |--------|
|  **xlnet-base** <br>(12 attention heads, 12 layers)  | from Pytorch-transformer |          |         |    5     |        
|  **xlnet-m1** <br>(12 attention heads, 16 layers)    | xlnet_m1.json            |          |         |    5     |
|  **xlnet-m2** <br>(12 attention heads, 16 layers) **RUNNING**  | xlnet_m2.json            |          |         |    5     |  
|  **xlnet-large** <br>(16 attention heads, 24 layers)  | from Pytorch-transformer |  0.056   |   5.430    |    5     |



## Working example

```python
from pytorch_transformers import *

```



## References
[1] #Bidaf [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603)
[2] [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) 
[3] [Stanford CS224n NLP Class w/Ashish Vaswani & Anna Huang](https://www.youtube.com/watch?v=5vcj8kSwBCY&t=76s)
