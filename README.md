# [Tito Joker](http://streamlit.thinkingmachin.es:8080/)
## Humorous AI model that uses state-of-the-art Deep Learning to tell jokes

This model, named **Tito Joker**, was built with the goal of creating an AI that understands humor well enough to tell jokes that are actually funny. He still has a long way to go but we will get there!

**Why is the model named Tito Joker?** Because in Filipino, "tito" means "uncle" when translated to English, and in the Philippines, we all have that uncle who says the corniest jokes!

**Interact with Tito Joker on this [website](http://streamlit.thinkingmachin.es:8080/).**

## Current features
1. **Text input box** to enter any custom prompt for your joke (question prompts are recommended)
    1. Sample input: *Why did the chicken cross the road?*
    2. Sample output: *Why did the chicken cross the road? To get to the other side.*
2. **Language model settings** that are configurable from the webpage
    1. Max token count for output
    2. Number of jokes to generate (max 20)
    3. GIF generation (set to "No" by default)
        1. This is currently done by returning a GIF from GIPHY based on an entity that is detected from the generated joke using named entity recognition (NER)
3. **Feature requests form** for any functionality that you want in future versions of Tito Joker
4. **Data tracking** for generated jokes which are logged into a csv and can be used to:
    1. Recycle jokes that have already been generated in the past
    2. Train a humor detector model once feedback is integrated into the app
5. **Model caching** which speeds up text generation after the first time the model is run

## Future features
1. **Feedback system** that will allow users to rate the jokes that are being generated
    1. Feedback will be stored and can be used to further improve the model (e.g. by optimizing for "funniness")
2. **Easy model shifting** that will allow users to toggle between different versions of the model at the web app level
    1. For example, if I want to visualize the difference between the results of GPT-2 vs DistilGPT-2, easy model shifting will make this much faster
3. **Joke type controls** that will give users the ability to specify the *type* of joke that they want to generate
    1. For example, we might want to explicitly tell Tito Joker to generate *Yo Mama* type of jokes
    2. This will require us to account for joke *type* at training time
4. **Context controls** that will allow users to feed Tito Joker *contextual information* that they want to use for the generated joke
    1. An idea would be to apply a similar specification with [BERT for Q&A](https://arxiv.org/abs/1810.04805), where both a question and context paragraphs are used as inputs to the model
    2. The effect is that the joke generated should be *about* the context paragraphs that are used as inputs

## Setup

```
# Clone the repo
git clone https://github.com/enzoampil/tito-joke-generator.git

# Navigate to the repo's directory
cd tito-joke-generator

# Create and activate a virtual environment
virtualenv venv
source venv/bin/activate

# Run the setup script which includes downloading the trained model and installing the necessary packages
sh setup.sh
```

## Run Tito Joker as a Streamlit webapp

```
streamlit run tito-jokes-app.py
```

## Methodology

### Architecture
Fine-tuned version of the recently released [OpenAI GPT-2 model](https://openai.com/blog/gpt-2-1-5b-release/) with a left-to-right language modeling training objective. Similar hyperparameters were used from the [original paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

### Data
A [jokes dataset](https://www.kaggle.com/abhinavmoudgil95/short-jokes) from Kaggle was used for fine-tuning. Aside from the original preprocessing, additional special tokens were added to allow the model to understand the difference between the "question" and "answer" components of a riddle type joke.

## Trained models
1. [Tito Joker v1 (OpenAI GPT-2)](https://storage.googleapis.com/joke-generator-model1/model1.zip)

## Acknowledgments

Special thanks to [Hugging Face](https://huggingface.co/) for their implementation of OpenAI GPT-2 using PyTorch and [Thinking Machines Data Science](https://thinkingmachin.es/) for sponsoring the server that I am running Tito Joker on.
