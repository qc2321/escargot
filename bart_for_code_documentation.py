import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers.generation import GenerationConfig
from transformers import Trainer, TrainingArguments
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import DataCollatorForSeq2Seq

df = pd.read_csv('./python-train_clean.tsv', sep='\t', names=['text', 'summary'])
dataset = Dataset.from_pandas(df)

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", attention_dropout=0.1)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def get_features(batch):
    input_encodings = tokenizer(batch["text"], max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(batch["summary"], max_length=256, truncation=True)

    return {"input_ids": input_encodings["input_ids"],
           "attention_mask": input_encodings["attention_mask"],
           "labels": target_encodings["input_ids"]}

dataset_ftrs = dataset.map(get_features, batched=True)
columns = ['input_ids', 'labels', 'input_ids','attention_mask',]
dataset_ftrs.set_format(type='torch', columns=columns)

training_args = TrainingArguments(
    output_dir='./models/bart-summarizer',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    # per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

model.config.output_attentions = True
model.config.output_hidden_states = True

training_args = TrainingArguments(
    output_dir='./models/bart-summarizer',
    num_train_epochs=1,
    warmup_steps=500,
    per_device_train_batch_size=1,
    # per_device_eval_batch_size=1,
    weight_decay=0.01,
    logging_steps=10,
    push_to_hub=False,
    # evaluation_strategy='steps',
    # eval_steps=500,
    save_steps=1e6,
    gradient_accumulation_steps=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=seq2seq_data_collator,
    train_dataset=dataset_ftrs,
    # eval_dataset=dataset_ftrs["test"],
)

trainer.train()

trainer.save_model('./bart_code_summ_ft.pt')

modelFT = BartForConditionalGeneration.from_pretrained('./bart_code_summ_ft.pt')

ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = modelFT.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

