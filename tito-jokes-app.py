import streamlit as st
import yaml
from run_generation import read_model_tokenizer, generate_text
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import os
from random import randrange
import spacy
import giphy_client
from giphy_client.rest import ApiException

CONFIG = "config.yaml"
SOURCE = "model1.zip"
TARGET = "model1.zip"
BUCKET = "joke-generator-model1"
JOKES_FP = "./data/jokes_generated.csv"
MAX_SAMPLES = 20
SEED_RANGE = 100000
DEFAULT_QUESTION = "Why did the chicken cross the road?"
GENERATE_MULTIPLE_OPTION = False
SEED_GENERATOR = True
GIF_GENERATOR = False
ABOUT_SECTION = """
&nbsp; &nbsp; &nbsp;
### About

This model, named "Tito Joker", was built with the goal of creating an AI that understands humor well enough to tell jokes that are actually funny. The code is fully open-sourced on [github](https://github.com/enzoampil/tito-joker).

It was trained by fine-tuning the recently released [OpenAI GPT-2](https://openai.com/blog/gpt-2-1-5b-release/) on an open sourced [jokes dataset](https://www.kaggle.com/abhinavmoudgil95/short-jokes).

Why is the model named Tito Joker? Because in Filipino, "tito" means "uncle" when translated to English, and in the Philippines, we all have that uncle who says the corniest jokes!

### Acknowledgments

Special thanks to [Hugging Face](https://huggingface.co/) for their implementation of OpenAI GPT-2 using PyTorch and [Thinking Machines Data Science](https://thinkingmachin.es/) for sponsoring the server that I am running Tito Joker on.

"""


class Struct:
    """
    Class to convert a dictionary into an object where values are accessible as atttributes
    """

    def __init__(self, **entries):
        self.__dict__.update(**entries)

    def __repr__(self):
        return "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])


def get_giphy(
    q,  # str | Search query term or prhase.
    api_key="dc6zaTOxFJmzC",  # str | Giphy API Key.
    limit=1,  # int | The maximum number of records to return. (optional) (default to 25)
    offset=0,  # int | An optional results offset. Defaults to 0. (optional) (default to 0)
    rating="g",  # str | Filters results by specified rating. (optional)
    lang="en",  # str | Specify default country for regional content; use a 2-letter ISO 639-1 country code. See list of supported languages <a href = \"../language-support\">here</a>. (optional)
    fmt="json",
):
    try:
        api_instance = giphy_client.DefaultApi()
        # Search Endpoint
        api_response = api_instance.gifs_search_get(
            api_key, q, limit=limit, offset=offset, rating=rating, lang=lang, fmt=fmt
        )
        results = api_response.to_dict()["data"]
        if results:
            first_result_id = results[0]["id"]
            return "https://media.giphy.com/media/{}/giphy.gif".format(first_result_id)
        else:
            return ""

    except ApiException as e:
        print("Exception when calling DefaultApi->gifs_search_get: %s\n" % e)
        return ""


def get_tokens(text, pos=["NOUN"]):
    """
    Return tokens with a specified list of pos tags
    """
    doc = nlp(text)
    return [t for t in doc if t.pos_ in pos]


def update_csv(df, fp, check_columns=False):
    # This should be in a separate module as a test
    # This is meant to avoid the error that occurs when a new column is added to the jokes file
    if check_columns:
        orig_df = pd.read_csv(fp)
        orig_num_cols = orig_df.shape[1]
        new_num_cols = df.shape[1]
        assert orig_num_cols == new_num_cols, 'Number of columns not equal (new - {}, orig - {}'.format(new_num_cols, orig_num_cols)
    # if file does not exist write header
    if not os.path.isfile(fp):
        print("File ({}) not found!")
        df.to_csv(fp, header="column_names", index=False)
        print("File ({}) created!")
    else:  # else it exists so append without writing the header
        print("File ({}) found!")
        df.to_csv(fp, mode="a", header=False, index=False)
        print("File ({}) updated!")

def backfill_csv(fp, funny, not_funny):
    """
    Replace 'funny' and 'not_funny' columns of the last row with the given values if both are None
    """
    df = pd.read_csv(fp)
    last_row = df.iloc[-1]
    if np.isnan(last_row['funny']) and np.isnan(last_row['not_funny']):
        print('Both "funny" and "not_funny" from the last row are None ...')
        print('Replacing with {} for "funny" and {} for "not_funny" ...'.format(funny, not_funny))
        last_row['funny'] = funny
        last_row['not_funny'] = not_funny
        df.iloc[-1] = last_row
        df.to_csv(fp, index=False)
        print('Last row replacement finished!')
    else:
        print('"funny" and "not_funny" from the last row are NOT both None ...')
        print('CSV was left untouched!')

def make_directory(directory):
    if not os.path.exists(directory):
        print("Directory ({}) not found!".format(directory))
        os.makedirs(directory)
        print("Directory ({}) created!".format(directory))


def read_yaml(fp):
    """
    Read yaml file and return as namedtuple (GeneratorConfig)
    """
    with open(fp) as f:
        y = yaml.load(f, Loader=yaml.FullLoader)

    gen_config = Struct(**y)
    return gen_config


def get_config(fp):
    args = read_yaml(fp)
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = torch.cuda.device_count()
    args.model_type = args.model_type.lower()
    return args


def clean_joke(joke):
    joke = joke.replace("<eoq>", "?").strip()
    return " ".join([t for t in joke.split() if t[0] != "<" and t[-1] != ">"])


@st.cache(allow_output_mutation=True)
def read_model_tokenizer_cached(model_type, model_name_or_path, device):
    model, tokenizer = read_model_tokenizer(model_type, model_name_or_path, device)
    return model, tokenizer


def tell_joke(args, begin):
    model, tokenizer = read_model_tokenizer_cached(args)
    # `begin` is only added to `args` after the model is read so that caching can be done properly (static input)
    args.prompt = begin.replace("?", "")
    jokes = generate_text(args, model, tokenizer)
    processed_jokes = [args.prompt + " " + joke for joke in jokes]
    return processed_jokes


if __name__ == "__main__":

    st.markdown(
        """
    # Ask [Tito Joker](https://github.com/enzoampil/tito-joker) anything! :)
    
    Created by [Lorenzo Ampil](https://medium.com/@lorenzo.ampil).
    
    """
    )

    st.sidebar.markdown("### Settings")

    num_tokens = st.sidebar.selectbox(
        "Max token count for output", [10, 20, 40, 80, 160], index=2
    )

    if GENERATE_MULTIPLE_OPTION:
        num_samples = st.sidebar.selectbox(
            "Number of jokes to generate", list(range(1, MAX_SAMPLES + 1)), index=0
        )
    else:
        num_samples = 1

    generate_gif = st.sidebar.selectbox(
        "Generate a GIF?", ["No", "Yes"], index=1 if GIF_GENERATOR else 0
    )

    st.sidebar.markdown(
        "Send feature requests [here](https://forms.gle/yuivvdpQxgRGoq238)."
    )

    begin = st.text_input("Ask any question", DEFAULT_QUESTION)

    # Add About section
    st.sidebar.markdown(ABOUT_SECTION)

    args = get_config(CONFIG)
    args.length = num_tokens
    args.num_samples = num_samples

    if SEED_GENERATOR:
        # This ensures that samples are unique, even with similar prompts
        args.seed = randrange(SEED_RANGE)

    print(args)
    current_timestamp = datetime.strftime(datetime.utcnow(), "%Y-%m-%dT%H:%M:%S")
    model, tokenizer = read_model_tokenizer_cached(
        args.model_type, args.model_name_or_path, args.device
    )
    args.prompt = begin.replace("?", "")
    jokes = generate_text(args, model, tokenizer)
    jokes = [args.prompt + " " + joke for joke in jokes]

    enumerated_jokes = [
        "### " + str(i + 1) + ". " + clean_joke(joke) for i, joke in enumerate(jokes)
    ]

    for i, joke in enumerate(enumerated_jokes):
        st.markdown(joke)
        funny_key = 'funny_{}'.format(i)
        not_funny_key = 'not_funny_{}'.format(i)
        funny = st.button('Funny', key=funny_key)
        not_funny = st.button('Not funny', key=not_funny_key)
        print("Feedback for previous joke: funny={}, not_funny={}".format(funny, not_funny))

    if generate_gif == "Yes":
        nlp = spacy.load("en_core_web_sm")
        # Return gif based on first common noun found (if at least 1 common noun found)
        jokes_nouns = get_tokens(jokes[0])
        if jokes_nouns:
            giphy_url = get_giphy(jokes_nouns[0].text)
            if giphy_url:
                print("Giphy url:", giphy_url)
                st.markdown("![Alt Text]({})".format(giphy_url))

    make_directory("data")
    # Split jokes in question & answer.
    split_jokes = pd.DataFrame([j.split("<eoq>") for j in jokes]).fillna("")

    # Only save the joke if at least one of the jokes have a complete question
    if split_jokes.shape[1] == 2:
        # Then, save as a csv.
        split_jokes.columns = ["question", "answer"]
        split_jokes["timestamp_utc"] = current_timestamp
        split_jokes["prompt"] = args.prompt

        print(split_jokes)
        split_jokes = split_jokes.applymap(clean_joke)
        split_jokes["funny"] = None
        split_jokes["not_funny"] = None

        # Backfill feedback to previous joke generated
        backfill_csv(JOKES_FP, funny, not_funny)

        # Update csv with new joke
        update_csv(split_jokes, JOKES_FP)
        print("Generated jokes updated to {} !".format(JOKES_FP))

