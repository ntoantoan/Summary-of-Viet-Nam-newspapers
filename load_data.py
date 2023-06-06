from datasets import Dataset
import pandas as pd
df_train = pd.read_csv("datasets/text_summarization_train.csv")
df_val = pd.read_csv("datasets/text_summarization_valid.csv")

input_lines = df_train['text']
label_lines = df_train['label']

print(df_train["label"][5])