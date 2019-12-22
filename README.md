# Hi, I am [Tito Joker](http://streamlit.thinkingmachin.es:8080/)! :wave: :grinning:
## A humorous AI model that uses state-of-the-art Deep Learning to tell jokes

**Tito Joker** is an AI that aims to understand humor well enough to tell jokes that are actually funny. All you have to do is input a riddle type question and he tells a joke using it. He still has a long way to go but we will get there!

![](typing.gif)

**Interact with Tito Joker on this [website](http://streamlit.thinkingmachin.es:8080/).**

*Warning: the training dataset contains NSFW jokes, so Tito Joker's humour will also reflect jokes of this nature.*

## Automatic GIF generation based on entities 
*Turned off by default - turn on from left sidebar*

![](gif_generator.gif)

## Ability to tell multiple jokes of up to 20 at a time
*Set this on left sidebar*

<img src="multiple.png" width="700">

## Methodology

### Architecture
Fine-tuned version of the recently released [OpenAI GPT-2 model](https://openai.com/blog/gpt-2-1-5b-release/) with a left-to-right language modeling training objective. Similar hyperparameters were used from the [original paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

### Data
A [jokes dataset](https://www.kaggle.com/abhinavmoudgil95/short-jokes) from Kaggle was used for fine-tuning. Aside from the original preprocessing, additional special tokens were added to allow the model to understand the difference between the "question" and "answer" components of a riddle type joke.

## Trained models
1. [Tito Joker v1 (OpenAI GPT-2)](https://storage.googleapis.com/joke-generator-model1/model1.zip)

## Acknowledgments

Special thanks to [Hugging Face](https://huggingface.co/) for their implementation of OpenAI GPT-2 using PyTorch and [Thinking Machines Data Science](https://thinkingmachin.es/) for sponsoring the server that I am running Tito Joker on.

## About

**Why is the model named Tito Joker?** Because in Filipino, "tito" means "uncle" when translated to English, and in the Philippines, we all have that uncle who says the corniest jokes!
