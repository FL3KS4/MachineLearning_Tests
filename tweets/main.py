import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import transformers as tr
from transformers import DistilBertTokenizerFast
from datasets import load_metric

# 1. Cleaning data

dataset_columns = ['target', 'id', 'date', 'flag', 'user', 'text']
dataset_encoding = "ISO-8859-1"
df = pd.read_csv('data/training.csv', encoding=dataset_encoding, names=dataset_columns)

data = df[['text', 'target']]

data['target'] = data['target'].replace(4, 1)

data_neg = data[data['target'] == 0]
data_neg = data_neg.sample(10000)
# data_pos.head()

data_pos = data[data['target'] == 1]
data_pos = data_pos.sample(10000)
# data_pos.head()

dataset = pd.concat([data_pos, data_neg])
dataset.head()

# 2. Making model

x = list(dataset['text'])
# print(x[:5])

y = list(dataset['target'])
# print(y[:5])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=69, stratify=y)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(x_train, truncation=True, padding=True)
test_encodings = tokenizer(x_test, truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
))

training_args = tr.TFTrainingArguments(
    output_dir='/results/result',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
)

with training_args.strategy.scope():
    model = tr.TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    temp = metric.compute(predictions=predictions, references=labels)
    print(temp)
    return temp


trainer = tr.TFTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()



# print & save model !!!

print(trainer.evaluate(test_dataset))

print(trainer.predict(test_dataset))



#trainer.save_model('Models/')
#loss = 0.393
#acc = 0.82325