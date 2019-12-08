import streamlit as st
import yaml
from run_generation import read_model_tokenizer, generate_text
import torch
import pandas as pd
from datetime import datetime
import os
from random import randrange

CONFIG = "config.yaml"
SOURCE = "model1.zip"
TARGET = "model1.zip"
BUCKET = "joke-generator-model1"
MAX_SAMPLES = 20
DEFAULT_QUESTION = 'Why did the chicken cross the road?'
SEED_GENERATOR = False

class Struct:
    """
    Class to convert a dictionary into an object where values are accessible as atttributes
    """
    def __init__(self, **entries):
        self.__dict__.update(**entries)
        
    def __repr__(self):
        return '\n'.join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])

def update_csv(df, fp):
    # if file does not exist write header 
    if not os.path.isfile(fp):
        print("File ({}) not found!")
        df.to_csv(fp, header='column_names', index=False)
        print("File ({}) created!")
    else: # else it exists so append without writing the header
        print("File ({}) found!")
        df.to_csv(fp, mode='a', header=False, index=False)
        print("File ({}) updated!")
       
def make_directory(directory):
    if not os.path.exists(directory):
        print('Directory ({}) not found!'.format(directory))
        os.makedirs(directory)
        print('Directory ({}) created!'.format(directory))

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
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.model_type = args.model_type.lower()
    return args

def clean_joke(joke):
    joke = joke.replace('<eoq>', '?').strip()
    return ' '.join([t for t in joke.split() if t[0] != '<' and t[-1] != '>'])

def tell_joke(args):
    model, tokenizer = read_model_tokenizer(args)
    jokes = generate_text(args, model, tokenizer)
    processed_jokes = [args.prompt + ' ' + joke for joke in jokes]
    return processed_jokes

if __name__=='__main__':

    st.markdown("""
    # Ask Tito Joker anything! :)
    
    Created by [Lorenzo Ampil](https://medium.com/@lorenzo.ampil).
    
    """)
    num_tokens = st.sidebar.selectbox(
        'Token count for output',
        [10, 20, 40, 80, 160],
        index=2)

    num_samples = st.sidebar.selectbox(
        'Number of jokes to generate',
        list(range(1, MAX_SAMPLES + 1)),
        index=0)
    
    st.sidebar.markdown("Send feature requests [here](https://forms.gle/yuivvdpQxgRGoq238).")

    begin = st.text_input('Ask any question', DEFAULT_QUESTION)
    
    args = get_config(CONFIG)
    args.prompt = begin.replace('?', '')
    args.length = num_tokens
    args.num_samples = num_samples
    
    if not SEED_GENERATOR:
        # This ensures that samples are unique, even with similar prompts
        args.seed = randrange(10000)
    
    print(args)
    current_timestamp = datetime.strftime(datetime.utcnow(), '%Y-%m-%dT%H:%M:%S')
    jokes = tell_joke(args)

    enumerated_jokes = [str(i + 1) + '. ' + clean_joke(joke) for i, joke in enumerate(jokes)]
    st.write('\n'.join(enumerated_jokes))
    
    make_directory('data')
    # Split jokes in question & answer.
    split_jokes = pd.DataFrame([j.split("<eoq>") for j in jokes]).fillna("")
    
    # Only save the joke if it's not the default question and at least one of the jokes have a complete question
    if split_jokes.shape[1] == 2:
        # Then, save as a csv.
        split_jokes.columns = ['question', 'answer']
        split_jokes['timestamp_utc'] = current_timestamp
        split_jokes['prompt'] = args.prompt
        print(split_jokes)
        split_jokes = split_jokes.applymap(clean_joke)
        split_jokes_fn = './data/jokes_generated.csv'
        update_csv(split_jokes, split_jokes_fn)        
        
        print('Generated jokes updated to {} !'.format(split_jokes_fn))