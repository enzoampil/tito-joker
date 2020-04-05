#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import click
import pandas as pd

@click.command()
@click.argument("raw_fp", type=click.Path(exists=True))
@click.argument("target_fp", type=click.Path(exists=True))
def process_jokes(raw_fp="shortjokes.csv", target_fp="riddle_jokes.txt"):
    df = pd.read_csv(raw_fp)

    # Append token at the end of each joke to indicate the end of a joke

    what_jokes = df[df.Joke.str.lower().str.startswith("what")].Joke.str.split("?")
    how_jokes = df[df.Joke.str.lower().str.startswith("how")].Joke.str.split("?")
    why_jokes = df[df.Joke.str.lower().str.startswith("why")].Joke.str.split("?")
    when_jokes = df[df.Joke.str.lower().str.startswith("when")].Joke.str.split("?")
    where_jokes = df[df.Joke.str.lower().str.startswith("where")].Joke.str.split("?")

    jokes = []
    for joke_ in [what_jokes, how_jokes, why_jokes, when_jokes, where_jokes]:
        joke_df_ = pd.DataFrame(joke_.values.tolist()).iloc[:, :2].dropna()
        joke_df_.columns = ["questions", "answer"]
        jokes.append(joke_df_)

    jokes_df = pd.concat(jokes)
    jokes_df = (
        jokes_df[~(jokes_df.answer.isin([""]))].drop_duplicates().reset_index(drop=True)
    )

    riddle_jokes_list = (
        "<soq> " + jokes_df.questions + " <eoq> " + jokes_df.answer + " <|endoftext|>"
    ).values.tolist()
    riddle_jokes = " <eoj> \n".join(riddle_jokes_list)

    with open(target_fp, "w") as f:
        f.write(riddle_jokes)