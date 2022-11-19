from nltk.translate.bleu_score import corpus_bleu
import nltk
nltk.data.path.append('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/zengweihao02/conda-env/seq2seq/nltk_data')      
def compute_bleus(sources, targets):
    """生成bleu分数，包括bleu-2, bleu-4
    Args:
        sources (List[str]): prediction, 注意不包含空格
        targets (List[str]): groundtruth, 注意不包含空格

    Returns:
        Dict[str: float]: bleu scores, {
            "bleu-2": 0.0,
            "bleu-4": 0.0
        }
    """
    assert len(sources) == len(targets), "Error in computing bleu scores because length of prediction doesn't equal to groundtruth"
    list_of_hypotheses = []
    list_of_references = [] 
    for i in range(len(sources)):
        if sources[i] and targets[i]:
            list_of_hypotheses.append(list(sources[i].strip()))  # 计算bleu时要手动按字切分
            list_of_references.append([list(targets[i].strip())])
        else:
            print("Warning: there is empty string when computing bleu scores")
    bleu2_weights = [0.5, 0.5]
    bleu4_weights = (0.25, 0.25, 0.25, 0.25)
    print("******Computing bleu scores******")
    return {
        "bleu-2": corpus_bleu(list_of_references, list_of_hypotheses, weights=bleu2_weights),
        "bleu-4": corpus_bleu(list_of_references, list_of_hypotheses, weights=bleu4_weights)  # main result
    }     
    