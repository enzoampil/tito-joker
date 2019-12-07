#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:42:43 2019

@author: enzoampil
"""

import pandas as pd

df = pd.read_csv('shortjokes.csv')

# Append token at the end of each joke to indicate the end of a joke

what_jokes = df[df.Joke.str.startswith('What')].Joke.str.split('?')
what_jokes.shape

how_jokes = df[df.Joke.str.startswith('How')].Joke.str.split('?')
how_jokes.shape

why_jokes = df[df.Joke.str.startswith('Why')].Joke.str.split('?')
why_jokes.shape

when_jokes = df[df.Joke.str.startswith('When')].Joke.str.split('?')
when_jokes.shape

jokes = []
for joke_ in [what_jokes, how_jokes, why_jokes, when_jokes]:
    joke_df_ = pd.DataFrame(joke_.values.tolist()).iloc[:, :2].dropna()
    joke_df_.columns = ['questions', 'answer']
    jokes.append(joke_df_)
    
jokes_df = pd.concat(jokes)
jokes_df = jokes_df[~(jokes_df.answer.isin(['']))].drop_duplicates().reset_index(drop=True)

riddle_jokes_list = ('<soq> ' + jokes_df.questions + ' <eoq> ' + jokes_df.answer + ' <eoa> ').values.tolist()

riddle_jokes = ' <eoj> \n'.join(riddle_jokes_list)


# Tokenize the joke string into the necessary format

with open('riddle_jokes.txt', 'w') as f:
    f.write(riddle_jokes)
