---
layout: post
title: "LLM Literature Review"
date: 2025-07-17 10:00:00 -0500
author: "Ben Johnson"
math: true
toc: true
reading_time: "15 min @ 250 wpm"
references:
---

This article is a literature review of LLM research between 2017-2025. The reader should be familiar with basic coding concepts, but does not need detailed knowledge of neural networks.

After reading this guide you should understand:

1. Why LLMs have become increasingly powerful in the last 3 years
2. What are the current research frontiers for LLMs

This article focuses on algorithmic research, but does not cover hardware innovations. LLM research has been made possible by access to large-scale compute, so it’s worth noting that hardware advancements are equally critical but will not be addressed in this guide.


## Neural Networks

A neural network, in its simplest form, is a function which maps inputs to outputs. The most basic building block is a linear transformation which uses weights **W** to map input vector **x** onto an output vector **y**. In practice, transformations are nonlinear and multiple layers of transformations are stacked together.

$$\hat{y} = Wx + b$$

The model is trained by predicting training data, calculating the distance of the predictions from the ground truth results, and updating the model weights. The distance of the predictions from the ground truth is called the **loss**, and the training objective is to minimize the **loss function**.

Prior to transformers, there were two dominant types of network architectures: recurrent neural networks (RNNs) and convolutional neural networks (CNNs).

### RNNs

**Recurrent neural networks** were designed to process sequential data, where order and timing matters. This includes temporal data and natural language. RNNs learn patterns over time by maintaining a hidden state which serves as the memory from previous states.

The sequential architecture of RNNs prevents the use of parallelization when training the model. Sequential computation was a major barrier to improving the quality of the model, as the time it took to train a model on a massive dataset would be prohibitively long.

### CNNs

**Convolutional neural networks** were designed to process grid-like data such as images and later adapted for natural language processing. CNNs use layers of convolutions to extract features from an input. A convolution is a mathematical operation where a small filter is passed over the input dataset, and at each position adjacent features are combined using a dot product.

CNNs process data in parallel, but they have trouble capturing long-range dependencies. Each layer operates over a small localized region of the dataset, so distant features must pass through many layers before they can be related. As the distance between two features grows, the depth and computational cost of the network increases.

### Transformers

Transformers solved both of these problems:
1. Transformers process data in parallel, unlike RNNs
2. Transformers process data independently of position

The main component in a transformer is attention. Attention decouples position from content, allowing the model to consider relationships between tokens regardless of their distance in the sequence. Attention also allows transformer networks to process input sequences in parallel, unlike RNNs which process tokens one at a time.

The original transformer architecture included two components: an **encoder** and a **decoder**. The encoder takes a sequence of text and transforms it into a sequence of vectors. The decoder takes the encoder output, along with the tokens generated so far, and predicts the next token in the sequence. The two architectures which emerged were:

**Encoder-only networks (BERT).** These models transform input text into a sequence of embeddings which captures the semantic meaning of the text. However, these models do not generate the next token in the sequence (they are not autoregressive) so they cannot be used for text generation. Instead their outputs are used for classification and retrieval.

**Decoder-only networks (GPT).** There is no encoder, so instead both the input and output text are treated as a single continuous sequence. The model predicts the next token in the sequence based on the previous tokens. These models are autoregressive and can be used to generate text or code.

## Training Methods

There are three different ways of training models:

**Supervised.** The model is trained on labeled data. For image classification, this would involve training the model on images which have been annotated. The major limitation with supervised learning is the size of the dataset - creating a labelled dataset is time intensive and costly so this training method is often limited by dataset size.

**Unsupervised.** Train the model on unlabeled data. In the case of language modeling, the way training is done is to mask the next word in a sequence and have the model predict the masked word. The model can then be trained to optimize this prediction. Unsupervised learning has the advantage of large dataset sizes, but there is no control over the data so it can be difficult to steer the model.

**Semi-supervised.** Use both unsupervised and supervised learning to train the model. GPT used semi-supervised learning to train the model. The major innovation was creating two steps to the training process: unsupervised pre-training and supervised fine-tuning.

The goal of unsupervised pre-training was to initialize the model with general knowledge which would help it adapt to more specific tasks. After pre-training the model can be fine tuned by training it on labelled datasets.

### Scaling Laws
This paper found that scaling the amount of compute, training data, and model parameters increased the performance of the model. Researchers found a relatively predictable relationship between performance and these three input variables, and were able to establish a way to predict model performance based on the amount of compute available, and optimize based on that compute.

This paper was the start of investing massive amounts of compute into training LLMs. For context, training GPT 4 required 10^25 FLOPS, or more compute than exists in the entire world.

## Reinforcement Learning

Reinforcement learning is the study of agents and how they learn by trial and error. RL trains an **agent** to make decisions by interacting with an **environment**, in order to maximize a **reward**. RL is an optimization problem where we try to find the conditions which allows the agent to achieve the maximum reward.

In pre-training, models are trained to predict the next word in a sequence. This can lead to undesirable outcomes - including sycophancy, where the model tells the user what they want to hear without regard to accuracy. It can also lead to models generating harmful text such as instructions for making a weapon.

Models are not aligned with human values, because the goal of the model is to predict the next most probable token. In order to train the model on a specific goal we need to apply another step of training, which is often RL. The training pipeline now includes two steps: generative pre-training (GPT) and post-training using reinforcement learning.


### Process Supervision

Cognitive scientists have described two types of thinking in humans: System 1 thinking, which is automatic and intuitive, and System 2 thinking, which is deliberate and reasoned.
LLMs initially exhibited results closer to System 1 thinking. The most obvious example is hallucinations, where the LLM generates a response which is factually inaccurate because it does not have an immediate answer to the user’s question.

Hallucinations are an example of an alignment problem: the model is trained to generate the next token in the sequence, not to generate an accurate response. Hallucinations initially prevented LLMs from solving complex math problems, because the models would generate a quick response without reasoning through mathematical constraints.

Reinforcement learning can develop reasoning capabilities by post-training the model on a labelled set of data with intermediate reasoning steps. The model is rewarded for generating a response which most closely matches human-labelled reasoning steps. Reasoning models such as o1 have been able to solve complex math and physics problems.

### Outcome Supervision

Labelled data with intermediate reasoning steps is expensive to produce. Most datasets are labelled at the outcome level rather than the process level. For example, test banks usually have a labelled answer key but do not have a labelled set of reasoning steps.

DeepSeek trained their R1 model without labelled process data, by post-training the model on outcome-labelled data. They saw dramatic improvements in model performance which were comparable to models trained with process supervision. DeepSeek demonstrated that outcome supervision is sufficient to train the model with reasoning capabilities.

This result is contested, as other research labs have found that process supervision outperforms outcome supervision. In theory, process supervision can search over a larger space than outcome supervision, and provides more clear feedback. It’s not clear which training method is better, as this is an active area of research with each lab testing a slightly different approach.

### Automated Process Supervision

Automated Process Supervision solves the data collection issue by automating the generation of process data. The basic approach involves generating a decision tree of potential solutions, and then evaluating the solutions up to the first incorrect step. This can then be used to label the accuracy of each path in the decision tree.

Automated data is noisy, because it is not always correct. This is one potential limitation with APS - the generated dataset includes incorrect data, which is then used to post-train the model. The other limitation is that APS does require some human supervision to label the outcome as correct or incorrect.

Process supervision labels data as correct or incorrect, but does not label the quality of the responses. Another limitation is the model will  “unlearn” some high-quality responses which are not the preferred response.  ByteDance is exploring a method which would add quality information to the dataset, and therefore avoids overconditioning on the golden dataset.

## Future Research

Autoregressive transformers have become well-established as the current neural network architecture powering language models. There is research into alternative architectures, but for now most research is focused on improving post-training methods to embed new capabilities in the models.