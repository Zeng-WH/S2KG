from nltk.util import ngrams
from nltk import word_tokenize
from collections import Counter, OrderedDict
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import numpy as np
import nltk
import math
#nltk.data.path.append('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/zengweihao02/conda-env/seq2seq/nltk_data')
class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, parallel_corpus):
        '''

        :param parallel_corpus: zip(list of str 1,list of str 2)
        :return: bleu4 score
        '''
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]
        empty_num = 0
        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            if hyps == ['']:
                empty_num += 1
                continue

            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]

            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        # print('empty turns:',empty_num)
        return bleu * 100
def eval_bleu(references, candidates):
    bleu_scorer=BLEUScorer()
    gen_ss, true_ss = [], []
    for ref, cand in zip(references, candidates):
        gen_ss.append([' '.join(list(ref))])
        true_ss.append([' '.join(list(cand))])
    bleu = bleu_scorer.score(zip(gen_ss, true_ss))
    return {
        "BLEU":bleu
    }
    
def compute_bleu(references, candidates):
    ref_list, dec_list = [], []
    for i in range(len(candidates)):
        dec_list.append(word_tokenize(candidates[i]))
        if type(references[i]) is list:
            tmp = []
            for ref in references[i]:
                tmp.append(word_tokenize(ref))
            ref_list.append(tmp)
        else:
            ref_list.append([word_tokenize(references[i])])
    bleu1 = corpus_bleu(ref_list, dec_list,
                        weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(ref_list, dec_list,
                        weights=(0, 1, 0, 0))
    bleu3 = corpus_bleu(ref_list, dec_list,
                        weights=(0, 0, 1, 0))
    bleu4 = corpus_bleu(ref_list, dec_list,
                        weights=(0, 0, 0, 1))
    return {
        "bleu-1": bleu1,
        "bleu-2": bleu2,
        "bleu-3": bleu3,
        "bleu-4": bleu4,  # main result
    } 
 
    
def compute_meteor(references, candidates):
    score_list = []
    ref_list, dec_list = [], []
    for i in range(len(candidates)):
        dec_list.append(word_tokenize(candidates[i]))
        if type(references[i]) is list:
            tmp =[]
            for ref in references[i]:
                tmp.append(word_tokenize(ref))
            ref_list.append(tmp)
            #ref_list = references[i]
        else:
            #ref_list = [references[i]]
            ref_list.append([word_tokenize(references[i])])
        score = meteor_score(ref_list[i], dec_list[i])
        score_list.append(score)
        
    return {
       "METEOR: ":  np.mean(score_list),
    }
    
    
def distinct_ngram(candidates, n=2):
    """Return basic ngram statistics, as well as a dict of all ngrams and their freqsuencies."""
    ngram_freqs = {}   # ngrams with frequencies
    ngram_len = 0  # total number of ngrams
    for candidate in candidates: 
        for ngram in ngrams(word_tokenize(candidate), n):
            ngram_freqs[ngram] = ngram_freqs.get(ngram, 0) + 1
            ngram_len += 1
    # number of unique ngrams
    uniq_ngrams = len([val for val in ngram_freqs.values() if val == 1])
    distinct_ngram = len(ngram_freqs) / ngram_len if ngram_len > 0 else 0
    print(f'Distinct {n}-grams:', round(distinct_ngram,4))
    return ngram_freqs, uniq_ngrams, ngram_len