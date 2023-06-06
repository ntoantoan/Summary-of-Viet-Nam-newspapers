import torch 
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_metric
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments

tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large-vietnews-summarization")  
model = AutoModelForSeq2SeqLM.from_pretrained("output/checkpoint-3017")
model.to("cuda")


metrics = load_metric("rouge")

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["inputs"], max_length=1024, truncation=True, padding=True
    )
    
    labels = tokenizer(
            examples["labels"], max_length=256, truncation=True, padding=True
        )
    model_inputs['labels'] = labels['input_ids']
    model_inputs['input_ids'] = model_inputs['input_ids']
    return model_inputs



df_val = pd.read_csv("datasets/text_summarization_valid.csv")
input_lines = df_val['text']
label_lines = df_val['label']
input_lines  = input_lines
label_lines = label_lines
dict_obj = {'inputs': input_lines, 'labels': label_lines}

dataset = Dataset.from_dict(dict_obj)
test_tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=4)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

max_target_length = 256
dataloader = torch.utils.data.DataLoader(test_tokenized_datasets, collate_fn=data_collator, batch_size=32)

predictions = []
references = []
for i, batch in enumerate(tqdm(dataloader)):
  outputs = model.generate(
      input_ids=batch['input_ids'].to('cuda'),
      max_length=max_target_length,
      attention_mask=batch['attention_mask'].to('cuda'),
  )
  with tokenizer.as_target_tokenizer():
    outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]

    labels = np.where(batch['labels'] != -100,  batch['labels'], tokenizer.pad_token_id)
    actuals = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in labels]
  predictions.extend(outputs)
  references.extend(actuals)
  metrics.add_batch(predictions=outputs, references=actuals)

results = metrics.compute()
print(results)