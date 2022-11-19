# SereTOD Track ReadMe

运行过程中出现问题欢迎联系：
ZengWH@bupt.edu.cn

系统分为用户意图分类，系统意图分类，系统回复生成三部分。请依次运行得到最终结果文件。



## 安装

```shell
pip install -r requirements.txt
```




## 用户意图分类

```shell
cd ./user_intent
bash test_intent_roberta_search_ensemble.sh
```

## 系统意图分类

```shell
cd ./system_intent
bash test_intent_roberta_search_sys_ensemble.sh
```

## 系统回复生成

```shell
cd ./response_gen
bash test_seq2seq_1B_submit.sh
```

## 提交文件生成

```shell
python submission.py
```


# Interact

交互式文件请使用

```shell
python3 interact.py
```

由于local KB在控制台上不太好输出，可以在interact.py中手动更改local_KB.

[](https://docs.google.com/spreadsheets/d/1w28AKkG6Wjmuo15QlRlRyrnv859MT1ry0CHV8tFxY9o/edit#gid=0)
