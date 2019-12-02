import streamlit as st
import subprocess
#from google.cloud import storage
import os

SOURCE = "model1.zip"
TARGET = "model1.zip"
BUCKET = "joke-generator-model1"
MAX_SAMPLES = 20
CACHE_RESULTS = False

def download_gcs(source, target, bucket_name):
    print('Downloading file "', source, '" from bucket: "', bucket_name, '"', 'to: "', target, '"')
    client = storage.Client.create_anonymous_client()
    # you need to set user_project to None for anonymous access
    # If not it will attempt to put egress bill on the project you specify,
    # and then you need to be authenticated to that project.
    bucket = client.bucket(bucket_name=bucket_name, user_project=None)
    blob = storage.Blob(source, bucket)
    blob.download_to_filename(filename=target, client=client)

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
        '''.format(num_tokens, num_samples, begin)

    out = subprocess.check_output(a, shell=True)
    jokes = (out.decode('utf-8').split('\n')[: -1])
    processed_jokes = [begin.replace('?', '') + ' ' + clean_joke(joke) for joke in jokes]
    return processed_jokes

if __name__=='__main__':
#    if not os.path.exists(url.name):
#        download_gcs(SOURCE, TARGET, TARGET)

    """
    # Ask Tito Joker anything! :)
    
    Created by [Lorenzo Ampil](https://www.linkedin.com/in/lorenzoampil/)
    """
    num_tokens = st.sidebar.selectbox(
        'Token count for output',
        [10, 20, 40, 80, 160],
        index=2)

    num_samples = st.sidebar.selectbox(
        'Number of jokes to generate',
        list(range(1, MAX_SAMPLES + 1)),
        index=0)

    begin = st.text_input('Ask any question', 'Why did the chicken cross the road?')

    jokes = tell_joke(begin, num_samples, num_tokens)
    enumerated_jokes = [str(i + 1) + '. ' + joke for i, joke in enumerate(jokes)]
    st.write('\n'.join(enumerated_jokes))