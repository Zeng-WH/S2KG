from transformers.trainer_utils import speed_metrics
from transformers.file_utils import is_torch_tpu_available
from transformers import T5Tokenizer
from transformers import MT5ForConditionalGeneration
import copy
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

def start_dialogue(tokenizer, model):
    local_KB = input('请输入local KB')
    
    local_KB = {
      "NA": {
        "用户需求": "我想包点流量"
      },
      "ent-1": {
        "name": "三十是五百兆",
        "type": "流量包",
        "业务费用": "三十",
        "流量范围": "五百兆"
      },
      "ent-2": {
        "name": "四十七白",
        "type": "流量包",
        "业务费用": "四十",
        "流量总量": "七白"
      },
      "ent-3": {
        "name": "五十一个g的,五十一个g,五十的",
        "type": "流量包",
        "业务费用": "五十",
        "流量总量": "一个g"
      },
      "ent-4": {
        "name": "十元一百",
        "type": "流量包",
        "业务费用": "十元",
        "流量总量": "一百"
      },
      "ent-5": {
        "name": "二十元三百兆",
        "type": "流量包",
        "业务费用": "二十元",
        "流量总量": "三百兆"
      },
      "ent-6": {
        "name": "七十元两个g,七十的,流量套餐",
        "type": "流量包",
        "业务费用": "七十,七十元",
        "流量总量": "两个g ,两个g",
        "流量范围": "国内"
      },
      "ent-7": {
        "name": "升级优惠包",
        "type": "流量包",
        "流量总量": "一个g",
        "流量范围": "省内"
      },
      "ent-8": {
        "name": "四g飞享,套餐",
        "type": "4g套餐",
        "业务费用": "八块钱",
        "流量总量": "一个g的省内,一个g的全国",
        "业务时长": "送十二个月"
      },
      "ent-9": {
        "name": "活动",
        "type": "业务",
        "办理渠道": "营业厅"
      }
    }
    KB_seq = convert_KB_to_sequences(local_KB)
    utter_list = []
    
    
    #start_user = input('请输入用户turn')
    
    while True:
        user_message = input('用户:')
        prev_utter_list = copy.deepcopy(utter_list)
        if user_message == 'quit':
            break
        else:
            utter_list.append(user_message)
            prev_u = prev_utter_list[-1:]
            dialog_his = '知识:'+KB_seq+' 用户0:' + ''.join(prev_u) +' ' + ' 用户:'+user_message
            dialog_inputids = tokenizer(dialog_his, return_tensors="pt").input_ids
            outputs = model.generate(dialog_inputids, num_beams=7, max_length=150)
            
            outputs_txt = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print("客服:", outputs_txt)
        
    
    
    

def main():
    nlg_tokenizer = T5Tokenizer.from_pretrained(
        'AndrewZeng/S2KG-base',
        extra_ids=0,
    )
    
    nlg_model = MT5ForConditionalGeneration.from_pretrained(
        'AndrewZeng/S2KG-base',
    )
    
    nlg_model.resize_token_embeddings(len(nlg_tokenizer))
    
    start_dialogue(nlg_tokenizer, nlg_model)
    
if __name__ == "__main__":
    main()    

    
    
    
'''
tokenizer = T5Tokenizer.from_pretrained(
        './response_gen/nlg_checkpoint/T5-1B/checkpoint-91512',
        extra_ids=0,
    )

model = MT5ForConditionalGeneration.from_pretrained(
        './response_gen/nlg_checkpoint/T5-1B/checkpoint-91512',
    )

model.resize_token_embeddings(len(tokenizer))

input_ids = tokenizer("知识: 用户0:北京话费  用户:北京话费是多少", return_tensors="pt").input_ids

outputs = model.generate(input_ids, num_beams=7, max_length=150)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

'''

