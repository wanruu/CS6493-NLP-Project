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


shot= "Context: This Main Building, and the library collection, was entirely destroyed by a fire in April 1879, and the school closed immediately and students were sent home. The university founder, Fr. Sorin and the president at the time, the Rev. William Corby, immediately planned for the rebuilding of the structure that had housed virtually the entire University. Construction was started on the 17th of May and by the incredible zeal of administrator and workers the building was completed before the fall semester of 1879. The library collection was also rebuilt and stayed housed in the new Main Building for years afterwards. Around the time of the fire, a music hall was opened. Eventually becoming known as Washington Hall, it hosted plays and musical acts put on by the school. By 1880, a science program was established at the university, and a Science Hall (today LaFortune Student Center) was built in 1883. The hall housed multiple classrooms and science labs needed for early research at the university. \n Based on the above context, generate the question for the following answer: Washington Hall \n Question: What was the music hall at Notre Dame called?\n\n Context: In 2015-2016, Notre Dame ranked 18th overall among \”national universities\” in the United States in U.S. News & World Report's Best Colleges 2016. In 2014, USA Today ranked Notre Dame 10th overall for American universities based on data from College Factual. Forbes.com's America's Best Colleges ranks Notre Dame 13th among colleges in the United States in 2015, 8th among Research Universities, and 1st in the Midwest. U.S. News & World Report also lists Notre Dame Law School as 22nd overall. BusinessWeek ranks Mendoza College of Business undergraduate school as 1st overall. It ranks the MBA program as 20th overall. The Philosophical Gourmet Report ranks Notre Dame's graduate philosophy program as 15th nationally, while ARCHITECT Magazine ranked the undergraduate architecture program as 12th nationally. Additionally, the study abroad program ranks sixth in highest participation percentage in the nation, with 57.6% of students choosing to study abroad in 17 countries. According to payscale.com, undergraduate alumni of University of Notre Dame have a mid-career median salary $110,000, making it the 24th highest among colleges and universities in the United States. The median starting salary of $55,300 ranked 58th in the same peer group.\nBased on the above context, generate the question for the following answer: 18th overall \n Question: Where did U.S. News & World Report rank Notre Dame in its 2015-2016 university rankings?"
@retry(wait=wait_random_exponential(min=8, max=50), stop=stop_after_attempt(6))


def gen(model,a):
    generate=openai.Completion.create(
    #model="text-davinci-003",
    model=model,
    prompt=a,
    max_tokens=800,
    stop=["Context:",'Q',"A"],
    temperature=0
)["choices"]
    print(generate[0]["text"])
    return(generate[0]["text"])


def method(data_context, data_question, data_source_sentence, model):
    result={}
    if model=="text-davinci-003":
        openai.api_key = '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'
    if model=="code-davinci-002":
        # openai.api_key ='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'
        openai.api_key ='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'
    result={}
    for _ in tqdm.tqdm(range(len(data_context))):
        context=data_context[_]['text']
        question=data_question[_]['text']
        source_sentence=data_source_sentence[_]['text']
        _id=str(_)
        #doc=row[3]
        a=shot+"Context: "+context+"\n\nQ: Based on the above context, generate the question for the answer in the source sentence:\n"+ source_sentence + "\n\nA:"
        raw=gen(model,a)
        result[_id] = {'raw': raw, 'prompt': a}

    return result


def save_exp(data_context, data_question, data_source_sentence, result, output):
    print(f'save results to {output}')
    init = (('context',[]),('question', []), ('answers', []), ('res',[]))
    save = OrderedDict(init)
    for _ in tqdm.tqdm(range(len(data_context))):
        context=data_context[_]['text']
        question=data_question[_]['text']
        source_sentence=data_source_sentence[_]['text']

        save['context'].append(context)
        save['question'].append(question)
        save['source_sentence'].append(source_sentence)
        save['res'].append(result[str(_)]['raw'])

    df = pd.DataFrame(data=save)
    df.to_csv(output)



if __name__ =='__main__':
    data_context = load_dataset(
        'text', data_files="./data/processed/para-dev.txt", split="train").select(range(0,1000))
    data_question = load_dataset(
        'text', data_files="./data/processed/tgt-dev.txt", split="train").select(range(0,1000))
    data_source_sentence = load_dataset(
        'text', data_files="./data/processed/src-dev.txt", split="train").select(range(0,1000))

    result=method(data_context, data_question, data_source_sentence, "code-davinci-002")
    save_exp(data_context, data_question, data_source_sentence, result,'squad_codex_res.csv')       

