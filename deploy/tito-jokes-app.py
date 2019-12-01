import streamlit as st
import subprocess

MAX_SAMPLES = 20
CACHE_RESULTS = False

"""# Ask Tito Joker anything! :)"""
num_tokens = st.sidebar.selectbox(
    'Token count for output',
     [10, 20, 40, 80, 160],
     index=2)

num_samples = st.sidebar.selectbox(
    'Number of candidate sequences to generate',
     list(range(1, MAX_SAMPLES + 1)),
     index=0)

begin = st.text_input('Question', 'Why did the chicken cross the road?')

def clean_joke(joke):
    return joke.replace('<soq>', '').replace('<eoa>', '').replace('<eoq>', '?').replace('<eoa', '')

@st.cache(persist=CACHE_RESULTS)
def tell_joke(begin, num_samples, num_tokens=40):
    a = '''python run_generation.py \
        --model_type=gpt2 \
        --model_name_or_path=./model1/ \
        --length={} \
        --num_samples={}\
        --stop_token="<eoj>"\
        --prompt="{}"
        '''.format(num_tokens, num_samples, begin.replace('?', ''))

    out = subprocess.check_output(a, shell=True)
    jokes = (out.decode('utf-8').split('\n')[: -1])
    processed_jokes = [begin + ' ' + clean_joke(joke) for joke in jokes]
    return processed_jokes

jokes = tell_joke(begin, num_samples, num_tokens)
enumerated_jokes = [str(i + 1) + '. ' + joke for i, joke in enumerate(jokes)]
st.write('\n'.join(enumerated_jokes))