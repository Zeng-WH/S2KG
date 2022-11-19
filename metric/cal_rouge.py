import rouge

# from run_seq2seq import main
rouge = rouge.Rouge()

def compute_rouges(sources, targets):
    """计算rouge-1、rouge-2、rouge-l

    Args:
        sources (List[str]): prediction, 注意不包含空格
        targets (List[str]): groundtruth, 注意不包含空格

    Returns:
        Dict[str: float]: rouge scores, {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0
        }
    """
    list_of_hypotheses = []
    list_of_references = []
    for i in range(len(sources)):
        if sources[i] and targets[i]:
            list_of_hypotheses.append(' '.join(sources[i].strip())) # 计算rouge时要手动在字间增加空格
            list_of_references.append(' '.join(targets[i].strip()))
            
        else:
            #print("Warning: there is empty string when computing rouge scores")
            pass
    try:
        scores = rouge.get_scores(list_of_hypotheses, list_of_references, avg=True)
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"]
        }
    except Exception:
        return {
            "rouge-1": 0,
            "rouge-2": 0,
            "rouge-l": 0
        }


if __name__ == "__main__":
    print(compute_rouges(["你好"], ["你好"]))