import streamlit as st
import pandas as pd
from datetime import datetime
from random import randrange
import spacy
from tito_joker import (
    get_config,
    read_model_tokenizer_cached,
    generate_text,
    clean_joke,
    get_tokens,
    get_giphy,
    make_directory,
    backfill_csv,
    update_csv,
)

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
JOKES_CSV_COLUMNS = [
    "question",
    "answer",
    "timestamp_utc",
    "prompt",
    "funny",
    "not_funny",
]
MODEL_VERSION_MAPPING = {
    "Tito Joker v1": "./model1/",
    "Tito Joker v2": "./model2/",
}

if __name__ == "__main__":

    st.markdown(
        """
    # Ask [Tito Joker](https://github.com/enzoampil/tito-joker) anything! :)
    
    Created by [Lorenzo Ampil](https://medium.com/@lorenzo.ampil).
    
    """
    )

    st.sidebar.markdown("### Settings")

    model_version = st.sidebar.selectbox(
        "Model version", ["Tito Joker v1", "Tito Joker v2"], index=1
    )

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
    args.model_name_or_path = MODEL_VERSION_MAPPING[model_version]

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
        funny_key = "funny_{}".format(i)
        not_funny_key = "not_funny_{}".format(i)
        funny = st.button("Funny", key=funny_key)
        not_funny = st.button("Not funny", key=not_funny_key)
        print(
            "Feedback for previous joke: funny={}, not_funny={}".format(
                funny, not_funny
            )
        )

    if generate_gif == "Yes":
        nlp = spacy.load("en_core_web_sm")
        # Return gif based on first common noun found (if at least 1 common noun found)
        jokes_nouns = get_tokens(jokes[0], nlp)
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
        backfill_csv(JOKES_FP, funny, not_funny, JOKES_CSV_COLUMNS)

        # Update csv with new joke
        update_csv(split_jokes, JOKES_FP)
        print("Generated jokes updated to {} !".format(JOKES_FP))
