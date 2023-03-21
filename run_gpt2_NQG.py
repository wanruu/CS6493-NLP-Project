from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from datasets import load_dataset
from collections import OrderedDict
import pandas as pd
import argparse
import tqdm

import config

shot = "Source sentence:the american football conference -lrb- afc -rrb- champion denver broncos defeated the national football conference -lrb- nfc -rrb- champion carolina panthers 24 -- 10 to earn their third super bowl title . \n" + \
    "So the question is:which nfl team represented the afc at super bowl 50 ?\n\n" + \
    "Source sentence:the game was played on february 7 , 2016 , at levi 's stadium in the san francisco bay area at santa clara , california . \n" + \
    "So the question is:where did super bowl 50 take place ?\n\n"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    default='gpt2',
    type=str,
    # required=True,
)
parser.add_argument(
    "--model_name_or_path",
    default='gpt2',
    type=str)

parser.add_argument("--prompt", type=str,
                    default='I don’t care if this is controversial,')
parser.add_argument("--length", type=int, default=128)
parser.add_argument("--stop_token", type=str, default=None,
                    help="Token at which text generation is stopped")

parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
)
parser.add_argument(
    "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
)
parser.add_argument("--k", type=int, default=0)
parser.add_argument("--p", type=float, default=0.9)

parser.add_argument("--prefix", type=str, default="",
                    help="Text added prior to input.")
parser.add_argument("--padding_text", type=str, default="",
                    help="Deprecated, the use of `--prefix` is preferred.")
parser.add_argument("--xlm_language", type=str, default="",
                    help="Optional language when used with the XLM model.")

parser.add_argument("--seed", type=int, default=42,
                    help="random seed for initialization")
parser.add_argument("--no_cuda", action="store_true",
                    help="Avoid using CUDA when available")
parser.add_argument("--num_return_sequences", type=int,
                    default=1, help="The number of samples to generate.")
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument("--load_adapter", type=str, default=None,
                    help="Path to a trained adapter")
parser.add_argument("--save_dir", type=str, default=None, help="Path to save")
parser.add_argument("--num", type=int, default=None,
                    help="num of sample times")

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = AutoModelForCausalLM.from_pretrained("gpt2")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.to(config.device)


data_question = load_dataset(
    'text', data_files="./data/processed/tgt-dev.txt", split="train")
data_source_sentence = load_dataset(
    'text', data_files="./data/processed/src-dev.txt", split="train")

result = {}
for _ in tqdm.tqdm(range(len(data_question))):
    question = data_question[_]["text"]
    source_sentence = data_source_sentence[_]["text"]
    prompt = shot + "Source sentence:" + source_sentence + "\nSo the question is:"
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(config.device)

    torch.manual_seed(0)
    outputs = model.generate(input_ids, do_sample=True, max_length=1024)
    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # find the third "So the question is:" in generated[0], because first two are in shot
    src, target = generated[0], "So the question is:"
    crop_idx = src.find(target) + len(target)  # first
    crop_idx += src[crop_idx:].find(target) + len(target)  # second
    crop_idx += src[crop_idx:].find(target) + len(target)  # third
    
    result[_] = {'generated question': generated[0][crop_idx:][0:300], 'gold question': question}


init = (('generated question', []), ('gold question', []))
save = OrderedDict(init)
for _ in tqdm.tqdm(range(len(data_question))):
    save['generated question'].append(result[_]['generated question'])
    save['gold question'].append(result[_]['gold question'])
df = pd.DataFrame(data=save)
df.to_csv(args.save_dir)
