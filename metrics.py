# BLEU-1,2,4, METEOR, ROUGE-L
import nltk
import numpy as np
from rouge import Rouge
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score


nltk.download("punkt")
nltk.download("wordnet")


class Metrics:
    def __init__(self, data):
        """
            data: [(generated, gold)] - [(str, str)]
        """
        self.generateds = [item[0] for item in data]
        self.golds = [item[1] for item in data]

        # tokenized
        self.token_generateds = [word_tokenize(g) for g in self.generateds]
        self.token_golds = [word_tokenize(g) for g in self.golds]

        # convert self.token_golds for bleu
        self.bleu_golds = [[g] for g in self.token_golds]

    @property
    def bleu_1(self):
        # TODO: this will be influenced by punctuations
        return corpus_bleu(self.bleu_golds, self.token_generateds, weights=(1, 0, 0, 0))
    
    @property
    def bleu_2(self):
        # TODO: this will be influenced by punctuations
        return corpus_bleu(self.bleu_golds, self.token_generateds, weights=(0.5, 0.5, 0, 0))
    
    @property
    def bleu_4(self):
        # TODO: this will be influenced by punctuations
        return corpus_bleu(self.bleu_golds, self.token_generateds, weights=(0.25, 0.25, 0.25, 0.25))
    
    @property
    def meteor(self):
        # TODO: this will be influenced by punctuations
        scores = [single_meteor_score(gold, generated) for generated, gold in zip(self.token_generateds, self.token_golds)]
        return np.mean(scores)

    @property
    def rouge_l(self):
        rouge = Rouge()
        score = rouge.get_scores(self.generateds, self.golds)[0]["rouge-l"]["f"]
        return score


if __name__ == "__main__":
    generated = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']
    gold = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']

    generated = " ".join(generated) + "."
    gold = " ".join(gold)

    M = Metrics([(generated, gold)])
    print(f"BLEU-1: {M.bleu_1}, BLEU-2: {M.bleu_2}, BLEU-4: {M.bleu_4}, METEOR: {M.meteor}, ROUGE-L: {M.rouge_l}")
