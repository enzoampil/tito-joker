# [Tito Joker](http://streamlit.thinkingmachin.es:8080/)
## Humorous AI model that uses state-of-the-art Deep Learning to tell jokes

This model, named **Tito Joker**, was built with the goal of creating an AI that understands humor well enough to tell jokes that are actually funny. He still has a long way to go but we will get there!

![](main.gif)

**Why is the model named Tito Joker?** Because in Filipino, "tito" means "uncle" when translated to English, and in the Philippines, we all have that uncle who says the corniest jokes!

**Interact with Tito Joker on this [website](http://streamlit.thinkingmachin.es:8080/).**

## Basic features
1. **Joke generation from any custom riddle input**
    1. Sample input: *Why did the chicken cross the road?*
    2. Sample output: *Why did the chicken cross the road? To get to the other side.*
2. **Automatic GIF generation based on entities found in the generated joke**
    1. Max token count for output
    2. Number of jokes to generate (max 20)
    3. GIF generation (set to "No" by default)
        1. This is currently done by returning a GIF from GIPHY based on an entity that is detected from the generated joke using named entity recognition (NER)
3. **Feature requests form** for any functionality that you want in future versions of Tito Joker

## Methodology

### Architecture
Fine-tuned version of the recently released [OpenAI GPT-2 model](https://openai.com/blog/gpt-2-1-5b-release/) with a left-to-right language modeling training objective. Similar hyperparameters were used from the [original paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

### Data
A [jokes dataset](https://www.kaggle.com/abhinavmoudgil95/short-jokes) from Kaggle was used for fine-tuning. Aside from the original preprocessing, additional special tokens were added to allow the model to understand the difference between the "question" and "answer" components of a riddle type joke.

## Trained models
1. [Tito Joker v1 (OpenAI GPT-2)](https://storage.googleapis.com/joke-generator-model1/model1.zip)

## Acknowledgments

Special thanks to [Hugging Face](https://huggingface.co/) for their implementation of OpenAI GPT-2 using PyTorch and [Thinking Machines Data Science](https://thinkingmachin.es/) for sponsoring the server that I am running Tito Joker on.
