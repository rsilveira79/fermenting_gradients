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

In this pandemic time, data-scientist and machine learning engineers are stepping in and building models to help policymakers take decisions under very uncertain moments. A great example is a friend of mine, Christian Perone, who is spending lots of energy to elucidate what is happening using Bayesian inference as the main tool. Check out his dedicated website including several different analysis: [Christian Perone - COVID-19 Analysis Repository](https://perone.github.io/covid19analysis/)

Christian pointed me one article from [Timothy W Russell](mailto:timothy.russell@lshtm.ac.uk) on how to estimate COVID-19 under-reporting using delay-adjusted case fatality ration. More details on Timothy's paper can be found here [1](#under_report).

Timothy has already provided a code in R [CFR calculation](https://github.com/thimotei/CFR_calculation) that can be used.
What I did here is to translate this code in R to Python, so that Pythonistas can use Pandas/Numpy/Scipy to perform the same calculations, or to replace the dataframe with data from your community.

## Method for estimating under-reporting

I'm not entering in detail about the method itself, if you need more information please refer to Timothy's paper [1](#under_report). Basically, dividing _deaths-to-date_ by _cases-to-date_ leads to a biased estimate of case fatality ratio (CFR), because this simple method does not account for delays from confirmation of a case to death, and under-reporting of cases.
This method adjusts the CFR by using using the distribution of the delay from _hospitalisation-to-death_, assuming that this delay is the same as _confirmation-to-death_. The distribution uses a _Lognormal_ fit, with a mean delay of 13 days and a standard deviation of 12.7 days.

The under-estimation can be calculated as:
$$u_{t}=\frac{\sum_{j=0}^{t}c_{t-j}f_{j}}{c_{t}}$$  
where:  
$$u_{t}$$ = underestimation of the proportion of cases with known outcomes  
$$c_{t}$$ = daily case incidence at time t  
$$f_{t}$$ = proportion of cases with delay of t between confirmation and death

For lognormal fit, I used scipy function `lognorm`:
```python
def plnorm(x, mu, sigma):
    shape  = sigma
    loc    = 0
    scale  = np.exp(mu)
    return lognorm.cdf(x, shape, loc, scale)
```
This will be used in the `hospitalisation_to_death_truncated` function, that is the delay function used in the adjustment:
```python
def hospitalisation_to_death_truncated(x,mu,sigma):
    return plnorm(x + 1, mu, sigma) - plnorm(x, mu, sigma)
```

The cCFR (corrected Case Fatality Ration) is calculated in this loop:
```python
cumulative_known_t = 0
for ii in range(0,len(df)):
    known_i = 0
    for jj in range(0,ii+1):
        known_jj = df['new_cases'].loc[ii-jj]*delay_func(jj)
        known_i = known_i + known_jj
    cumulative_known_t = cumulative_known_t + known_i
cum_known_t = round(cumulative_known_t)
nCFR = df['new_deaths'].sum()/df['new_cases'].sum()
cCFR = df['new_deaths'].sum()/cum_known_t
```
where `delay_func` is the `hospitalisation_to_death_truncated` speficied above (with **low**, **mid** and **high** mean and standard deviations).

The code can be found in this Jupyter Notebook here:

[COVID 19 - Under-reporting estimation](https://github.com/rsilveira79/CFR_calculation_python/blob/master/notebooks/1.initial_assessment.ipynb){: .btn .btn--success}

Also, I configured a Github Action to execute this code every 12 hours and output the *.CSV* file as result in *output* folder of this repo.

## References

1. **Using a delay-adjusted case fatality ratio to estimate under-reporting** <a name="under_report">[Link](https://cmmid.github.io/topics/covid19/global_cfr_estimates.html)</a><br>Timothy W Russell*, Joel Hellewell1, Sam Abbott1, Nick Golding, Hamish Gibbs, Christopher I Jarvis, Kevin van Zandvoort, CMMID nCov working group, Stefan Flasche, Rosalind Eggo, W John Edmunds & Adam J Kucharski, 2020
2. 


