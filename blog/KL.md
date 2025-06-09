*written on 06/08/2025* 

I was looking at the [KL estimation implemented by TRL](https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L510}) lately. 
It puzzles me because it does **not** seem to implement the *k3* estimation from [John Schulman's blog](http://joschu.net/blog/kl-approx.html).

The KL divergence between two probabilistic distributions $q$ and $p$ is written as:
$$
KL[q, p] =  \mathbb{E}_{x \sim q}[\log(\frac{q(x)}{p(x})]
$$

It can be approximated by Monte-Carlo (MC) sampling (numbered as *k1* in John Schulman's blog):
$$
KL[q, p] \approx  \frac{1}{N} \sum_x \log(\frac{q(x)}{p(x})
$$

For language models (LM), each distribution is a distribution over sequences, i.e., $x=\{x_i\}_{i=1}^T$. Therefore, the MC approximation becomes:
$$
KL[q, p] \approx  \frac{1}{N} \sum_x \log(\frac{\prod_i q(x_i|x_{<i})}{\prod_i p(x_i|x_{<i})})
$$

which can be broken down into steps:

$$
KL[q, p] \approx  \frac{1}{N} \sum_x [\log(\prod_i q(x_i|x_{<i})) - \log(\prod_i p(x_i|x_{<i}))]
 \approx \frac{1}{N} \sum_x \sum_i [\log q(x_i|x_{<i}) - \log p(x_i|x_{<i})]
$$

So, when implementing this, we can first calculate the stepwise log-probability difference $\log q(x_i|x_{<i}) - \log p(x_i|x_{<i})$ and then aggregate across $T$ steps. 

MC approximation is unbiased but it can have high variance, as it’s negative for half of the samples, whereas KL is always positive. In John Schulman's blog, they propose an alternative approximation (numbered as *k3*), which is unbiased and low-variance. 

$$
KL[q, p] \approx  \mathbb{E}_{x \sim q}[\log(\frac{q(x)}{p(x}) + \frac{p(x)}{q(x)} - 1]
\approx \frac{1}{N} \sum_x [\log(\frac{q(x)}{p(x}) + \frac{p(x)}{q(x)} - 1]
$$

This estimation is unbiased because:
$$
\mathbb{E}_{x \sim q}[\frac{p(x)}{q(x)} - 1] = \sum_x q(x) (\frac{p(x)}{q(x)} - 1) = \sum_x p(x) - \sum_x q(x) = 1 - 1 = 0
$$

However, this *k3* can **not** be broken down into steps in the context of LM: 

$$
KL[q, p]  \approx \frac{1}{N} \sum_x [\log(\frac{q(x)}{p(x}) + \frac{p(x)}{q(x)} - 1] 
 \approx \frac{1}{N} \sum_x \{ \sum_i [\log q(x_i|x_{<i}) - \log p(x_i|x_{<i})] + \frac{\prod_i p(x_i|x_{<i})}{\prod_i q(x_i|x_{<i})} - 1 \} (Eq.1)
$$

Due to the production ($\frac{\prod_i p(x_i|x_{<i})}{\prod_i q(x_i|x_{<i})}$), there is no way to write it as a sum over steps. However, if I understand correctly, what is implemented in TRL is:

$$
KL[q, p] \approx \frac{1}{N} \sum_x \sum_i [\log q(x_i|x_{<i}) - \log p(x_i|x_{<i}) + \frac{p(x_i|x_{<i})}{q(x_i|x_{<i})} - 1] \label{trl} (Eq.2)
$$

which is not equivalent to Eq.1. And this estimation is *biased* because:
$$
\mathbb{E}_{x \sim q}\{\sum_i [\frac{p(x_i|x_{<i})}{q(x_i|x_{<i})} - 1]\} = \sum_x \{ \prod_i q(x_i|x_{<i}) \sum_i [\frac{p(x_i|x_{<i})}{q(x_i|x_{<i})} - 1] \} \neq 0
$$

Nonetheless, I would not say Eq.2 is wrong. Eq.2 is nicely **non-negative**. And being **biased** is not necessarily bad. As mentioned in John Schulman's blog, they also sometimes use another estimator (numbered as *k2*) which is biased but **non-negative**. 

Sometimes I also feel puzzled by the notion of estimating KL divergence. Why it has to be KL divergence? What is so special of KL that makes us use it in various RL algorithms? I feel like it just needs to be some kind of divergence between two distributions. 
Maybe there are some more fundamental reasons that I don't yet grasp on.  

