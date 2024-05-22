
from transformers import AutoModel, AutoTokenizer
import logging
import pandas as pd
import datasets
from torchkeras import KerasModel
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModel, TrainingArguments, AutoConfig
import torch
from tqdm import tqdm
import transformers
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import warnings
from peft import PeftModel
model_name = "chatglm2-6b"
l_trainSet='train.csv'
l_devSet='dev.csv'
l_testSet='test.csv'
l_log='loging.log'
train_batch_size=18
dev_batch_size=18
l_dsize=300
l_epoch=200
l_patience=10
toxic_weight='toxic_chatglm6b'

tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()

prompt = """
你是一个专业的文本内容审核员。任务是对文本进行分类，分成“有毒”言论或者“无毒”言论。
下面是一些范例:
安徽还是挺穷的 -> 有毒
台湾人这么说 -> 无毒
请对下述评论进行分类。注意：只返回'有毒'或者'无毒'，无需其它说明和解释。
text ->
"""
def get_prompt(text):
    return prompt.replace('text', text)
response, his = model.chat(tokenizer, get_prompt('它们的大脑平滑到可以在上面溜冰，真的不是一个物种啊姐妹'), history=[])

his.append(("我来写关于我和她真实的六年故事吧。 -> ", "无毒"))
his.append(("你本来不就是少女 -> ", "无毒"))

his.append(("这点我同意，前提是不要要女性生孩子，男性也做一半家务，完美 -> ", "有毒"))
his.append(("看完了，我真的震惊，这种男人太恶心了。 -> ", "有毒"))
logging.basicConfig(filename=l_log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def dateLabelChange(date_path):
    import pandas as pd
    df = pd.read_csv(date_path)
    df['tag'] = df['toxic'].map({0: '无毒', 1: '有毒'})
    df = df.rename({'content': 'text'}, axis=1)
    return df
dateLabelChange(l_trainSet).to_parquet('dftrain.parquet')
dateLabelChange(l_devSet).to_parquet('dftest.parquet')

dftrain = pd.read_parquet('dftrain.parquet')
dftest = pd.read_parquet('dftest.parquet')
def build_inputs(query, history):
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
    prompt += "[Round {}]\n\n问：{} -> \n\n答：".format(len(history) + 1, query)
    return prompt

print(build_inputs('是重男轻女造成的恶果', history=his))

dftrain['context'] = [build_inputs(x, history=his) for x in dftrain['text']]
dftrain['target'] = [x for x in dftrain['tag']]
dftrain = dftrain[['context', 'target']]

dftest['context'] = [build_inputs(x, history=his) for x in dftest['text']]
dftest['target'] = [x for x in dftest['tag']]
dftest = dftest[['context', 'target']]

ds_train = datasets.Dataset.from_pandas(dftrain) # 转化为dataset格式
ds_val = datasets.Dataset.from_pandas(dftest)



model_name = model_name
max_seq_length = 512
skip_over_length = True

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True)

config = transformers.AutoConfig.from_pretrained(
    model_name, trust_remote_code=True, device_map='auto')
def preprocess(example):
    context = example["context"]
    target = example["target"]

    context_ids = tokenizer.encode(
        context,
        max_length=max_seq_length,
        truncation=True)

    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)

    input_ids = context_ids + target_ids + [config.eos_token_id]
    labels = [-100] * len(context_ids) + target_ids + [config.eos_token_id]

    return {"input_ids": input_ids,
            "labels": labels,
            "context_len": len(context_ids),
            'target_len': len(target_ids) + 1}


ds_train_token = ds_train.map(preprocess).select_columns(['input_ids', 'labels', 'context_len', 'target_len'])
if skip_over_length:
    ds_train_token = ds_train_token.filter(
        lambda example: example["context_len"] < max_seq_length and example["target_len"] < max_seq_length)
ds_val_token = ds_val.map(preprocess).select_columns(['input_ids', 'labels', 'context_len', 'target_len'])
if skip_over_length:
    ds_val_token = ds_val_token.filter(
        lambda example: example["context_len"] < max_seq_length and example["target_len"] < max_seq_length)

def data_collator(examples: list):
    len_ids = [len(example["input_ids"]) for example in examples]
    longest = max(len_ids)

    input_ids = []
    labels_list = []

    for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        ids = example["input_ids"]
        labs = example["labels"]

        ids = ids + [tokenizer.pad_token_id] * (longest - length)
        labs = labs + [-100] * (longest - length)

        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labs))

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

dl_train = torch.utils.data.DataLoader(ds_train_token, num_workers=2, batch_size=train_batch_size,
                                       pin_memory=True, shuffle=True,
                                       collate_fn=data_collator)
dl_val = torch.utils.data.DataLoader(ds_val_token, num_workers=2, batch_size=dev_batch_size,
                                     pin_memory=True, shuffle=True,
                                     collate_fn=data_collator)

dl_train.size = l_dsize


warnings.filterwarnings("ignore")
model = AutoModel.from_pretrained(model_name,
                                  load_in_8bit=False,
                                  trust_remote_code=True)

model.supports_gradient_checkpointing = True
model.gradient_checkpointing_enable()

model.config.use_cache = False

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32, lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)
model.is_parallelizable = True
model.model_parallel = True

model.print_trainable_parameters()

class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        if self.stage == 'train':
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch):

        # loss
        with self.accelerator.autocast():
            loss = self.net(input_ids=batch["input_ids"], labels=batch["labels"]).loss

        # backward()
        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        all_loss = self.accelerator.gather(loss).sum()
        step_losses = {self.stage + "_loss": all_loss.item()}
        step_metrics = {}

        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses, step_metrics

KerasModel.StepRunner = StepRunner

def save_ckpt(self, ckpt_path='checkpoint', accelerator=None):
    unwrap_net = accelerator.unwrap_model(self.net)
    unwrap_net.save_pretrained(ckpt_path)

def load_ckpt(self, ckpt_path='checkpoint'):
    import os
    self.net.load_state_dict(
        torch.load(os.path.join(ckpt_path, 'adapter_model.bin')), strict=False)
    self.from_scratch = False

KerasModel.save_ckpt = save_ckpt
KerasModel.load_ckpt = load_ckpt
keras_model = KerasModel(model, loss_fn=None,
                         optimizer=torch.optim.AdamW(model.parameters(), lr=2e-6))
ckpt_path = toxic_weight
keras_model.fit(train_data=dl_train,
                val_data=dl_val,
                epochs=l_epoch,
                patience=l_patience,  # 用于早停策略的参数。如果验证损失在连续的？个epochs中没有改善，则训练将提前停止。
                monitor='val_loss',  # 表示模型将监测验证损失。
                mode='min',  # 指定了性能监测的模式。
                ckpt_path=ckpt_path,
                mixed_precision='fp16'
                )

model = AutoModel.from_pretrained(model_name,
                                  load_in_8bit=False,
                                  trust_remote_code=True,
                                  device_map='auto')
model = PeftModel.from_pretrained(model, ckpt_path)
model = model.merge_and_unload()

def predict(text):
    response, history = model.chat(tokenizer, f"{text} -> ", history=his,temperature=0.01)
    return response

import csv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

csv_file_path = l_testSet
test_texts = []
test_labels = []
with open(csv_file_path, 'r', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        label = row['toxic']
        test_labels.append(int(label))
        text = row['content']
        test_texts.append(text)
prelabel_list = []

d_num = 0
for text in tqdm(test_texts):
    re = predict(text)
    if '无毒' in re:
        prelabel_list.append(0)
    else:
        prelabel_list.append(1)
    if len(re) > 2:
        d_num += 1

print("Precision:", precision_score(test_labels, prelabel_list))
print("Recall:", recall_score(test_labels, prelabel_list))
print("F1 score:", f1_score(test_labels, prelabel_list))
print("Acc", accuracy_score(test_labels, prelabel_list))
print(d_num)