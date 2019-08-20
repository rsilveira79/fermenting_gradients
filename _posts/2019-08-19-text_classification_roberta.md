---
layout: post
title:  "Text classification with RoBERTa"
date:   2019-08-19 18:21:41 -0300
comments: true 
image:
  path: /images/bert.jpeg
categories: machine_learning nlp pytorch
---

# Fine-tuning pytorch-transformers for SequenceClassification

As mentioned already in earlier [post](https://rsilveira79.github.io/fermenting_gradients/machine_learning/nlp/pytorch/pytorch-transformer-squad/), I'm a big fan of the work that the [Hugging Face](http://huggingface.co) is doing to make available latest models to the community.
Very recently, they made available Facebook RoBERTa: _A Robustly Optimized BERT Pretraining Approach_<sup> [1](#roberta)</sup>. Facebook team proposed several improvements on top of BERT  <sup> [2](#bert)</sup>, with the main assumption tha BERT model was _"significantly undertrained"_. The modification over BERT include:

1. training the model longer, with bigger batches;  
2. removing the next sentence prediction objective; 
3. training on longer sequences; 
4. dynamically changing the masking pattern applied to the training data;

More details can be found in the [paper](#bert), we will focus here on a practical application of RoBERTa model using `pytorch-transformers`library: text classification. 
For this practical application, we are going to use the SNIPs NLU (Natural Language Understanding) dataset <sup> [3](#snips)</sup>.

## NLU Dataset

The NLU dataset is composed by several intents, for this post we are going to use `2017-06-custom-intent-engines` dataset, that is composed by 7 classes:

* **SearchCreativeWork** (e.g. Find me the I, Robot television show);   
* **GetWeather** (e.g. Is it windy in Boston, MA right now?); 
* **BookRestaurant** (e.g. I want to book a highly rated restaurant for me and my boyfriend tomorrow night);
* **PlayMusic** (e.g. Play the last track from Beyoncé off Spotify);  
* **AddToPlaylist** (e.g. Add Diamonds to my roadtrip playlist);  
* **RateBook** (e.g. Give 6 stars to Of Mice and Men);  
* **SearchScreeningEvent** (e.g. Check the showtimes for Wonder Woman in Paris); 

## pytorch-transformers `RobertaForSequenceClassification`

As described in earlier [post](https://rsilveira79.github.io/fermenting_gradients/machine_learning/nlp/pytorch/pytorch-transformer-squad/), `pytorch-transormers` base their API in some main classes, and here it wasn't different:

* RobertaConfig
* RobertaTokenizer
* RobertaModel

All the code on this post can be found in this Colab notebook:  
[Text Classification with RoBERTa](https://colab.research.google.com/drive/1xg4UMQmXjDik3v9w-dAsk4kq7dXX_0Fm){: .btn .btn--success}

First things first, we need to import RoBERTa from `pytorch-transformers`, making sure that we are using latest release **<font color='red'>1.1.0</font>**:

```python
from pytorch_transformers import RobertaModel, RobertaTokenizer
from pytorch_transformers import RobertaForSequenceClassification, RobertaConfig

config = RobertaConfig.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification(config)
```

As the NLU dataset has 7 classes (labels), we need to set this in the RoBERTa configuration:

```python
config.num_labels = len(list(label_to_ix.values()))
```

```json
{
  "attention_probs_dropout_prob": 0.1,
  "finetuning_task": null,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 514,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "num_labels": 7,
  "output_attentions": false,
  "output_hidden_states": false,
  "torchscript": false,
  "type_vocab_size": 1,
  "vocab_size": 50265
}
```

In this notebook, I used the nice Colab GPU feature, so all the boilerplate code with `.cuda()` is there. Make sure you have the correct device specified [`cpu`, `cuda`] to run the train the classifier.

I fine-tuned the classifier for **3** epochs, using `learning_rate`= **1e-05**, with `Adam` optimizer and `nn.CrossEntropyLoss()`. Depending on the dataset you are dealing, these parameters need to be changed. After the **3** epochs, the train accuracy was **~ 98%**, which is fine considering a small dataset (and probably a bit of overfitting as well). 

Here are some results I got using the fine-tuned model with `RobertaForSequenceClassification`:

```python
get_reply("play radiohead song")
'PlayMusic'

get_reply("it is rainy in Sao Paulo")
'GetWeather'

get_reply("Book tacos for me tonight")
'BookRestaurant'

get_reply("Book a table for me tonight")
'BookRestaurant'
```

RoBERTo hopes you have enjoyed RoBERTa :smiley: and you can use it in your projects! 

## References

1. **RoBERTa: A Robustly Optimized BERT Pretraining Approach** <a name="roberta">[PDF](https://arxiv.org/abs/1907.11692)</a><br>Yinhan Liu and Myle Ott and Naman Goyal and Jingfei Du and Mandar Joshi and Danqi Chen and Omer Levy and Mike Lewis and Luke Zettlemoyer and Veselin Stoyanov, 2019
2. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** <a name="bert">[PDF](https://arxiv.org/abs/1810.04805)</a><br>Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova, 2018 
3. **Natural Language Understanding benchmark** <a name="snips">[Link](https://github.com/snipsco/nlu-benchmark)</a><br>Alice Coucke, Alaa Saade, Adrien Ball, Théodore Bluche, Alexandre Caulier, David Leroy, Clément Doumouro, Thibault Gisselbrecht, Francesco Caltagirone, Thibaut Lavril, Maël Primet, Joseph Dureau, 2018 

