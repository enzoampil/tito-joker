import streamlit as st
import yaml
from run_generation import read_model_tokenizer, generate_text
import torch
import pandas as pd
import numpy as np
import os
import giphy_client
from giphy_client.rest import ApiException


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


def get_tokens(text, nlp, pos=["NOUN"]):
    """
    Return tokens with a specified list of pos tags
    """
    doc = nlp(text)
    return [t for t in doc if t.pos_ in pos]


def create_empty_csv(fp, columns):
    empty_df = pd.DataFrame(columns=columns)
    print("Saving empty csv with columns:", columns)
    empty_df.to_csv(fp)
    print("Empty csv was saved to:", fp)


def update_csv(df, fp, check_columns=False):
    # This should be in a separate module as a test
    # This is meant to avoid the error that occurs when a new column is added to the jokes file
    if check_columns:
        orig_df = pd.read_csv(fp)
        orig_num_cols = orig_df.shape[1]
        new_num_cols = df.shape[1]
        assert (
            orig_num_cols == new_num_cols
        ), "Number of columns not equal (new - {}, orig - {}".format(
            new_num_cols, orig_num_cols
        )
    # if file does not exist write header
    if not os.path.isfile(fp):
        print("File ({}) not found!")
        df.to_csv(fp, header="column_names", index=False)
        print("File ({}) created!")
    else:  # else it exists so append without writing the header
        print("File ({}) found!")
        df.to_csv(fp, mode="a", header=False, index=False)
        print("File ({}) updated!")


def backfill_csv(fp, funny, not_funny, columns):
    """
    Replace 'funny' and 'not_funny' columns of the last row with the given values if both are None
    """
    if not os.path.isfile(fp):
        create_empty_csv(fp, columns)
    df = pd.read_csv(fp)
    last_row = df.iloc[-1]
    if np.isnan(last_row["funny"]) and np.isnan(last_row["not_funny"]):
        print('Both "funny" and "not_funny" from the last row are None ...')
        print(
            'Replacing with {} for "funny" and {} for "not_funny" ...'.format(
                funny, not_funny
            )
        )
        last_row["funny"] = funny
        last_row["not_funny"] = not_funny
        df.iloc[-1] = last_row
        df.to_csv(fp, index=False)
        print("Last row replacement finished!")
    else:
        print('"funny" and "not_funny" from the last row are NOT both None ...')
        print("CSV was left untouched!")


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
