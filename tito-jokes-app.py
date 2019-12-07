import streamlit as st
import yaml
from run_generation import read_model_tokenizer, generate_text
import torch

CONFIG = "config.yaml"
SOURCE = "model1.zip"
TARGET = "model1.zip"
BUCKET = "joke-generator-model1"
MAX_SAMPLES = 20

class Struct:
    """
    Class to convert a dictionary into an object where values are accessible as atttributes
    """
    def __init__(self, **entries):
        self.__dict__.update(**entries)
        
    def __repr__(self):
        return '\n'.join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])
        
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
    joke = joke.replace('<eoq>', '?')
    return ' '.join([t for t in joke.split() if t[0] != '<' and t[-1] != '>'])

@st.cache(allow_output_mutation=True)
def tell_joke(args):
    model, tokenizer = read_model_tokenizer(args)
    jokes = generate_text(args, model, tokenizer)
    processed_jokes = [args.prompt + ' ' + joke for joke in jokes]
    return processed_jokes

if __name__=='__main__':

    st.markdown("""
    # Ask Tito Joker anything! :)
    
    Created by [Lorenzo Ampil](https://medium.com/@lorenzo.ampil)
    """)
    num_tokens = st.sidebar.selectbox(
        'Token count for output',
        [10, 20, 40, 80, 160],
        index=2)

    num_samples = st.sidebar.selectbox(
        'Number of jokes to generate',
        list(range(1, MAX_SAMPLES + 1)),
        index=0)

    begin = st.text_input('Ask any question', 'Why did the chicken cross the road?')
    begin = begin.replace('?', '')
    
    args = get_config(CONFIG)
    args.prompt = begin
    args.num_tokens = num_tokens
    args.num_samples = num_samples
    print(args)

    jokes = tell_joke(args)
    enumerated_jokes = [str(i + 1) + '. ' + clean_joke(joke) for i, joke in enumerate(jokes)]
    st.write('\n'.join(enumerated_jokes))