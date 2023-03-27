# BLEU-1,2,4, METEOR, ROUGE-L
import os
import csv
import nltk
import numpy as np
from rouge import Rouge
# from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score


# nltk.download("punkt")
nltk.download("wordnet")


class Metrics:
    def __init__(self, generateds, golds):
        """
            data: [(generated, gold)] - [(str, str)]
        """
        self.generateds = generateds
        self.golds = golds

        # tokenized data, for bleu and meteor
        tokenizer = RegexpTokenizer(r"\w+")
        self.token_generateds = [tokenizer.tokenize(g) for g in generateds]
        self.token_golds = [tokenizer.tokenize(g) for g in golds]

        # convert self.token_golds format for bleu
        # bleu need reference list
        self.bleu_golds = [[g] for g in self.token_golds]

    @property
    def bleu_1(self):
        return corpus_bleu(self.bleu_golds, self.token_generateds, weights=(1, 0, 0, 0))
    
    @property
    def bleu_2(self):
        return corpus_bleu(self.bleu_golds, self.token_generateds, weights=(0.5, 0.5, 0, 0))
    
    @property
    def bleu_4(self):
        return corpus_bleu(self.bleu_golds, self.token_generateds, weights=(0.25, 0.25, 0.25, 0.25))
    
    @property
    def meteor(self):
        scores = [single_meteor_score(gold, generated) for generated, gold in zip(self.token_generateds, self.token_golds)]
        return np.mean(scores)

    @property
    def rouge_l(self):
        rouge = Rouge()
        score = rouge.get_scores(self.generateds, self.golds)[0]["rouge-l"]["f"]
        return score

    @property
    def text(self):
        _text = f"BLEU-1: {self.bleu_1:.3f}, BLEU-2: {self.bleu_2:.3f}, BLEU-4: {self.bleu_4:.3f}, " + \
            f"METEOR: {self.meteor:.3f}, ROUGE-L: {self.rouge_l:.3f}"
        return _text


def read_results_from(filenames, headers):
    generated_header, gold_header = headers[0], headers[1]
    generateds, golds = [], []
    for filename in filenames:
        with open(filename, encoding="utf-8") as f:
            for res in csv.DictReader(f):
                generateds.append(res[generated_header])
                golds.append(res[gold_header])
    return generateds, golds



if __name__ == "__main__":
    # For test
    # generated = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']
    # gold = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']

    # generated = " ".join(generated) + "."
    # gold = " ".join(gold)

    # M = Metrics([(generated, gold)])
    # print(f"BLEU-1: {M.bleu_1}, BLEU-2: {M.bleu_2}, BLEU-4: {M.bleu_4}, METEOR: {M.meteor}, ROUGE-L: {M.rouge_l}")

    # codex squad
    codex_squad_path = "res/codex_squad_res/"
    codex_squad_filenames = os.listdir(codex_squad_path)
    codex_squad_filenames = [codex_squad_path + f for f in codex_squad_filenames]
    codex_squad_headers = ["res", "question"]
    generateds, golds = read_results_from(codex_squad_filenames, codex_squad_headers)
    M = Metrics(generateds, golds)
    print("codex on squad:", M.text)

    # codex nqg
    codex_nqg_path = "res/codex_nqg_res/"
    codex_nqg_filenames = os.listdir(codex_nqg_path)
    codex_nqg_filenames = [codex_nqg_path + f for f in codex_nqg_filenames]
    codex_nqg_headers = ["res", "question"]
    generateds, golds = read_results_from(codex_nqg_filenames, codex_nqg_headers)
    M = Metrics(generateds, golds)
    print("codex on squad-nqg:", M.text)

    # gpt2 squad
    gpt2_squad_filename = "res/20230309_17_54gpt2.squad.csv"
    gpt2_squad_headers = ["generated question", "gold question"]
    generateds, golds = read_results_from([gpt2_squad_filename], gpt2_squad_headers)
    for g_idx, g in enumerate(generateds):
        idxs = set([g.find(","), g.find("."), g.find("?"), g.find("\n")]) - {-1}
        if idxs:
            generateds[g_idx] = g[:min(idxs)]
    M = Metrics(generateds, golds)
    print("gpt2 on squad:", M.text)

    # gpt2 nqg
    gpt2_nqg_path = "res/gpt2_nqg_res/"
    gpt2_nqg_filenames = os.listdir(gpt2_nqg_path)
    gpt2_nqg_filenames = [gpt2_nqg_path + f for f in gpt2_nqg_filenames]
    gpt2_nqg_headers = ["generated question", "gold question"]
    generateds, golds = read_results_from(codex_nqg_filenames, codex_nqg_headers)
    for g_idx, g in enumerate(generateds):
        idxs = set([g.find(","), g.find("."), g.find("?"), g.find("\n")]) - {-1}
        if idxs:
            generateds[g_idx] = g[:min(idxs)]
    M = Metrics(generateds, golds)
    print("gpt2 on squad-nqg:", M.text)

    # flant5 squad
    flant5_squad_filename = "res/20230326_17_53flant5.squad.csv"
    flant5_squad_headers = ["generated question", "gold question"]
    generateds, golds = read_results_from([flant5_squad_filename], flant5_squad_headers)
    M = Metrics(generateds, golds)
    print("flant5 on squad:", M.text)

    # flant5 nqg
    flant5_nqg_filename = "res/20230327_12_06flant5nqg.csv"
    flant5_nqg_headers = ["generated question", "gold question"]
    generateds, golds = read_results_from([flant5_nqg_filename], flant5_nqg_headers)
    M = Metrics(generateds, golds)
    print("flant5 on nqg:", M.text)
