import readline
import sklearn.metrics
import jsonlines


def compute_acc(sources, targets):
    """计算acc
    Args:
        sources (List[str]): prediction, 注意不包含空格
        targets (List[str]): groundtruth, 注意不包含空格

    Returns:
        Dict[str: float]: acc scores, {
            "acc": 0.0
        }
    """
    assert len(sources) == len(targets), "Error in computing acc scores because length of prediction doesn't equal to groundtruth"
    return {
        "acc": 100*sklearn.metrics.accuracy_score(targets, sources)
    }
  
if __name__ == "__main__":
    import tensorflow as tf
    import json
    # 计算mt5的分类任务acc
    pred_file = "viewfs://hadoop-meituan/user/hadoop-aipnlp/hekeqing/clue_finetune/t5-0.3B-finetune-pair-e8-d8-seq-const_lr1e-4-kefu-multitask-3960000-bs64-epoch50/evaluate.json"
    dev_file = "/home/hadoop-aipnlp/dolphinfs/hekeqing/projects/UFA/data/pair/dev.json"
    
    pred = [json.loads(line.strip())["label"] for line in tf.io.gfile.GFile(pred_file, "r").readlines()]
    reference = []
    with open(dev_file, 'r') as r:
        for item in jsonlines.Reader(r):
            reference.append(item["label"])
    print("acc=", compute_acc(pred, reference))
    print("macro f1", 100.0 * sklearn.metrics.f1_score(reference, pred, average="macro"))
    print("macro p", 100.0 * sklearn.metrics.precision_score(reference, pred, average="macro"))
    print("macro r", 100.0 * sklearn.metrics.recall_score(reference, pred, average="macro"))