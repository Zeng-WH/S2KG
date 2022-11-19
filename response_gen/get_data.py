import json

import copy
import tqdm

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

with open("./data/test_data2 for track2.json", "r") as r:
    label_train_json = json.load(r)

    
#with open("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/SereTODTrack2/SereTOD2022-main/Track2/PreTrain/nlg_data/datav1.1_all_KB/processed_data.json", "r") as r:
 #   unlabel_train_json = json.load(r)
    
all_train_json = []
all_train_json.extend(label_train_json)
#all_train_json.extend(unlabel_train_json)

example_list = []
count = 0
for item in tqdm.tqdm(all_train_json):
    utter_list = []
    content = item['content']
    KB = item['KB']
    KB_seq = convert_KB_to_sequences(KB)
    #print("bupt")
    for c in content:
        prev_utter_list = copy.deepcopy(utter_list)
        utter_list.append(c['用户'])
        #utter_list.append(c['客服'])
        temp_json = {}
        
        ppppppprev_u = prev_utter_list[-7:-6]
        pppppprev_u = prev_utter_list[-6:-5]
        ppppprev_u = prev_utter_list[-5:-4]
        pppprev_u = prev_utter_list[-4:-3]
        ppprev_u = prev_utter_list[-3:-2]
        pprev_u = prev_utter_list[-2:-1]
        prev_u = prev_utter_list[-1:]
        #temp_dialog = '知识:'+KB_seq+' 用户0:' + ''.join(prev_u) + ' ' + ' 客服0:' + ''.join(prev_s) + ' ' + ' 用户:'+c['用户']
        temp_dialog = '知识:'+KB_seq+' 用户0:' + ''.join(prev_u) +' ' + ' 用户:'+c['用户']
        #temp_dialog = '知识:'+KB_seq+' 用户0:' + ''.join(pprev_u)  + ' 用户1:' + ''.join(prev_u)+' 用户:'+c['用户']
        #temp_dialog = '知识:'+KB_seq+' 用户0:' + ''.join(ppprev_u)  + ' 用户1:' + ''.join(pprev_u)+ ' 用户2:' + ''.join(prev_u)+' 用户:'+c['用户']
        #temp_dialog = '知识:'+KB_seq+' 用户0:' + ''.join(pppprev_u)  + ' 用户1:' + ''.join(ppprev_u)+ ' 用户2:' + ''.join(pprev_u)+' 用户3:' + ''.join(prev_u)+' 用户:'+c['用户']
        #temp_dialog = '知识:'+KB_seq+' 用户0:' + ''.join(ppppprev_u)  + ' 用户1:' + ''.join(pppprev_u)+ ' 用户2:' + ''.join(ppprev_u)+' 用户3:' + ''.join(pprev_u)+' 用户4:' + ''.join(prev_u)+' 用户:'+c['用户']
        #temp_dialog = '知识:'+KB_seq+' 用户0:' + ''.join(pppppprev_u)  + ' 用户1:' + ''.join(ppppprev_u)+ ' 用户2:' + ''.join(pppprev_u)+' 用户3:' + ''.join(ppprev_u)+' 用户4:' + ''.join(pprev_u)+' 用户5:' + ''.join(prev_u)+' 用户:'+c['用户']
        temp_response = ' '
        temp_json['dialogue'] = temp_dialog
        if len(temp_dialog) > 1024:
            count += 1
        temp_json['response'] = temp_response
        example_list.append(temp_json)
with open("./data/test.json", "w", encoding='utf-8') as w:
    for item in example_list:
        w.write(json.dumps(item, ensure_ascii=False))   
        w.write("\n")
    
