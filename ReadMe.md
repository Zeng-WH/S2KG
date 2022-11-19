# S2KG ReadMe

**Semi-Supervised Knowledge-Grounded Pre-training for Task-Oriented Dialog Systems**

We present our models for Track 2 of the SereTOD 2022 challenge, which is the first challenge of building semi-supervised and reinforced TOD systems on a large-scale real-world Chinese TOD dataset MobileCS. We build a knowledge-grounded dialog model, S2KG to formulate dialog history and local KB as input and predict the system response.

[This paper](https://arxiv.org/abs/2210.08873) has been accepted at the SereTOD 2022 Workshop, EMNLP 2022

## System Performance
Our system achieves the first place both in the automatic evaluation and human interaction, especially with higher BLEU (+7.64) and Success (+13.6%) than the second place. The evaluation results for both Track 1 and Track 2, which can be accessed via this [this link](https://docs.google.com/spreadsheets/d/1w28AKkG6Wjmuo15QlRlRyrnv859MT1ry0CHV8tFxY9o/edit#gid=0).

## 🔥 News

- [**2022-11-19**]: We release our S2KG-base model for knowledge-grounded dialogue generation !
- [**2022-10-10**]: S2KG has been accepted at the SereTOD 2022 Workshop, EMNLP 2022 !
- [**2022-09-10**]: 🏆 Achieved the 1st rank on SereTOD 2022 track 2 !

## Model List
We released the S2KG-base model for knowledge-grounded dialogue generation. You can import this by using [HuggingFace's Transformers](https://github.com/huggingface/transformers).

| Model                                                        |
| ------------------------------------------------------------ |
| [AndrewZeng/S2KG-base](https://huggingface.co/AndrewZeng/S2KG-base) |


## Setup environment

```shell
pip install -r requirements.txt
```

## Quick Start

```python
from transformers.trainer_utils import speed_metrics
from transformers.file_utils import is_torch_tpu_available
from transformers import T5Tokenizer
from transformers import MT5ForConditionalGeneration

def convert_KB_to_sequences(KB):
    temp_seq = []
    for e in KB:
        if e == 'NA':
            NA_temp = []
            for na in KB['NA']:
                NA_temp.append(na+'：'+KB['NA'][na])
            temp_seq.append('；'.join(NA_temp))
        else:
            ent_info = KB[e]
            ent_temp = []
            for ent in ent_info:
                if ent == 'name':
                    ent_temp.append('名称：'+ent_info[ent])
                elif ent == 'type':
                    ent_temp.append('类型：'+ent_info[ent])
                else:
                    ent_temp.append(ent+'：'+ent_info[ent])
            temp_seq.append('；'.join(ent_temp))
    temp_seq = ' '.join(temp_seq)
    return temp_seq
    
local_KB = {
      "NA": {
        "欠费": "欠费,欠了_八块二毛八,八块多,八块二毛八",
        "持有套餐": "五十八的,它,活动"
      },
      "ent-1": {
        "name": "五十八的",
        "type": "业务",
        "业务费用": "五十八"
      },
      "ent-2": {
        "name": "它,活动",
        "type": "业务",
        "业务规则": "返了二十五块钱话费,每月返的二十五"
      }
    }
    
nlg_tokenizer = T5Tokenizer.from_pretrained(
        'AndrewZeng/S2KG-base',
        extra_ids=0,
    )  
nlg_model = MT5ForConditionalGeneration.from_pretrained(
        'AndrewZeng/S2KG-base',
    )
nlg_model.resize_token_embeddings(len(nlg_tokenizer))
KB_seq = convert_KB_to_sequences(local_KB)
dialog_his = '知识:'+KB_seq+' 用户0:' + '你好！' +' ' + ' 用户:'+'欠了多少钱？'
dialog_inputids = tokenizer(dialog_his, return_tensors="pt").input_ids
outputs = model.generate(dialog_inputids, num_beams=7, max_length=150)
outputs_txt = tokenizer.decode(outputs[0], skip_special_tokens=True)

```




## Reproduce Results

You can reproduce our response generation results by:

```shell
cd ./response_gen
bash test_seq2seq_1B_submit.sh
```

## Interact

You can interact with system using:

```shell
python3 interact.py
```

You can modify local KB in `interact.py` manually.


