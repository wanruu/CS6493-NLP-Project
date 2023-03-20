import math
import argparse
import pandas as pd
import openai
from time import sleep
import time
import os
from datasets import load_dataset
import nltk
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from collections import OrderedDict
import tqdm


shot= "Source sentence: the american football conference -lrb- afc -rrb- champion denver broncos defeated the national football conference -lrb- nfc -rrb- champion carolina panthers 24 -- 10 to earn their third super bowl title . \n Question: which nfl team represented the afc at super bowl 50 ?\n\nSource sentence: the game was played on february 7 , 2016 , at levi 's stadium in the san francisco bay area at santa clara , california . \n Question: where did super bowl 50 take place ?\n\n"
@retry(wait=wait_random_exponential(min=8, max=50), stop=stop_after_attempt(6))


def gen(model,a):
    generate=openai.Completion.create(
    #model="text-davinci-003",
    model=model,
    prompt=a,
    max_tokens=800,
    stop=['Q',"A"],
    temperature=0
)["choices"]
    print(generate[0]["text"])
    return(generate[0]["text"])


def method(data_question, data_source_sentence, model):
    result={}
    if model=="text-davinci-003":
        openai.api_key = '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'
    if model=="code-davinci-002":
        # openai.api_key ='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'
        openai.api_key ='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'
    result={}
    for _ in tqdm.tqdm(range(len(data_source_sentence))):
        question=data_question[_]['text']
        source_sentence=data_source_sentence[_]['text']
        _id=str(_)
        #doc=row[3]
        a=shot+"Q: Source sentence: "+ source_sentence + "\n\nA:"
        raw=gen(model,a)
        result[_id] = {'raw': raw, 'prompt': a}

    return result


def save_exp(data_question, data_source_sentence, result, output):
    print(f'save results to {output}')
    init = (('question', []), ('answers', []), ('res',[]))
    save = OrderedDict(init)
    for _ in tqdm.tqdm(range(len(data_source_sentence))):
        question=data_question[_]['text']
        source_sentence=data_source_sentence[_]['text']

        save['question'].append(question)
        save['source_sentence'].append(source_sentence)
        save['res'].append(result[str(_)]['raw'])

    df = pd.DataFrame(data=save)
    df.to_csv(output)



if __name__ =='__main__':
    data_question = load_dataset(
        'text', data_files="./data/processed/tgt-dev.txt", split="train").select(range(0,1000))
    data_source_sentence = load_dataset(
        'text', data_files="./data/processed/src-dev.txt", split="train").select(range(0,1000))

    result=method(data_question, data_source_sentence, "code-davinci-002")
    save_exp(data_question, data_source_sentence, result,'squad_codex_res.csv')       

