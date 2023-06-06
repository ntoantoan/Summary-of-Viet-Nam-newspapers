import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization")  
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base-vietnews-summarization")
model.to('cuda')



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


df_train = pd.read_csv("datasets/text_summarization_train.csv")
df_val = pd.read_csv("datasets/text_summarization_valid.csv")

input_lines = df_train['text']
label_lines = df_train['label']

dict_obj = {'inputs': input_lines, 'labels': label_lines}
dataset = Dataset.from_dict(dict_obj)

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=4)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")


training_args = Seq2SeqTrainingArguments(output_dir="outputs",
                                      do_train=True,
                                      do_eval=False,
                                      num_train_epochs=1,
                                      learning_rate=1e-5,
                                      warmup_ratio=0.05,
                                      weight_decay=0.01,
                                      per_device_train_batch_size=2,
                                      per_device_eval_batch_size=2,
                                      logging_dir='log',
                                      group_by_length=True,
                                      save_strategy="epoch",
                                      save_total_limit=3,
                                      #eval_steps=1,
                                      #evaluation_strategy="steps",
                                      # evaluation_strategy="no",
                                      fp16=True,
                                      )
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

trainer.train()
