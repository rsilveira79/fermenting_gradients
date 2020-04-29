---
layout: post
title:  "COVID 19 Under-reporting estimation"
date:   2020-04-19 15:00:00 -0300
comments: true 
image:
  path: /images/iceberg.jpg
categories: stats under-reporting covid
---

# COVID-19 - Under-Reporting estimation in Python

In this pandemic time, data-scientist and machine learning engineers are stepping in and building models to help policy makers take decisions under very uncertain moments. A great example is a friend of mine, Christian Perone, who is spending lots of energy in order to elucidade what is happening using bayesian inference as main tool. Check out his dedicated website including several different analysis: [Christian Perone - COVID-19 Analysis Repository](https://perone.github.io/covid19analysis/)

Christian pointed me one article from [Timothy W Russell](timothy.russell@lshtm.ac.uk) on how to estimate COVID-19 under-reporting using delay-adjusted case fatality ration. More details on Timothy's paper can be found here [1](#under_report).







As mentioned already in earlier [post](https://rsilveira79.github.io/fermenting_gradients/machine_learning/nlp/pytorch/pytorch-transformer-squad/), I'm a big fan of the work that the [Hugging Face](http://huggingface.co) is doing to make available latest models to the community.
Very recently, they made available Facebook RoBERTa: _A Robustly Optimized BERT Pretraining Approach_<sup> [1](#roberta)</sup>. Facebook team proposed several improvements on top of BERT  <sup> [2](#bert)</sup>, with the main assumption tha BERT model was _"significantly undertrained"_. The modification over BERT include:

1. training the model longer, with bigger batches;  
2. removing the next sentence prediction objective; 
3. training on longer sequences; 
4. dynamically changing the masking pattern applied to the training data;

More details can be found in the [paper](#bert), we will focus here on a practical application of RoBERTa model using `pytorch-transformers`library: text classification. 
For this practical application, we are going to use the SNIPs NLU (Natural Language Understanding) dataset <sup> [3](#snips)</sup>.


## pytorch-transformers `RobertaForSequenceClassification`

As described in earlier [post](https://rsilveira79.github.io/fermenting_gradients/machine_learning/nlp/pytorch/pytorch-transformer-squad/), `pytorch-transormers` base their API in some main classes, and here it wasn't different:

* RobertaConfig
* RobertaTokenizer
* RobertaModel

All the code on this post can be found in this Colab notebook:  
[Text Classification with RoBERTa](https://colab.research.google.com/drive/1xg4UMQmXjDik3v9w-dAsk4kq7dXX_0Fm){: .btn .btn--success}



## References

1. **Using a delay-adjusted case fatality ratio to estimate under-reporting** <a name="under_report">[Link](https://cmmid.github.io/topics/covid19/global_cfr_estimates.html)</a><br>Timothy W Russell*, Joel Hellewell1, Sam Abbott1, Nick Golding, Hamish Gibbs, Christopher I Jarvis, Kevin van Zandvoort, CMMID nCov working group, Stefan Flasche, Rosalind Eggo, W John Edmunds & Adam J Kucharski, 2020


