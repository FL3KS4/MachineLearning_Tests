from tensorflow import keras
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

dataset_columns = ['target', 'id', 'date', 'flag', 'user', 'text']
dataset_encoding = "ISO-8859-1"
df = pd.read_csv('data/training.csv', encoding=dataset_encoding, names=dataset_columns)

data = df[['target', 'text']]

data['target'] = data['target'].replace(4, 1)

data_neg = data[data['target'] == 0]
data_neg = data_neg.sample(300000)

data_pos = data[data['target'] == 1]
data_pos = data_pos.sample(300000)

dataset = pd.concat([data_pos, data_neg])
"""print(dataset.head())"""
dataset = shuffle(dataset)

test_neg3 = data[data['target'] == 0]
test_neg3 = test_neg3.sample(25000)

test_pos3 = data[data['target'] == 1]
test_pos3 = test_pos3.sample(25000)

dataset3 = pd.concat([test_pos3, test_neg3])
dataset3 = shuffle(dataset3)

"""Tokenizer"""
myTokenizer = Tokenizer(num_words=120000)
myTokenizer.fit_on_texts(dataset.iloc[:, 1])

train_data = myTokenizer.texts_to_sequences(dataset.iloc[:, 1])
test_data = myTokenizer.texts_to_sequences(dataset3.iloc[:, 1])

word_index = myTokenizer.word_index
word_index["<PAD>"] = 0

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=200)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=200)


x_train = train_data
y_train = dataset.iloc[:,0]

model = keras.Sequential()

model.add(keras.layers.Embedding(120000,16))
model.add(keras.layers.GlobalAvgPool1D())
model.add(keras.layers.Dense(16,activation="relu"))
model.add(keras.layers.Dense(8,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))

model.summary()

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss=tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])


fitmodel = model.fit(x_train, y_train, epochs=20, batch_size=768, verbose=1)

results = model.evaluate(test_data,dataset3.iloc[:,0])
print(results)

model.save("TweetsModel3rd3.h5")


def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded


#model = keras.models.load_model("TweetsModel3rd.h5")



y_pred=model.predict(test_data)

accuracy_sc = accuracy_score(y_pred=y_pred.round(),y_true=dataset3.iloc[:,0])*100


print("Accuracy score is {}% ".format(accuracy_sc))


def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment');
  plt.show()


class_names = ['negative', 'positive']
cm = confusion_matrix(dataset3.iloc[:,0], y_pred.round())
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

# Prediction on real-life examples
"""
with open("test.txt") as f:
    for line in f:
        nline = line.replace(",", "").replace(".","").replace("(","").replace(")","").replace(":","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        result = predict[0] * 100
        print(str(result) + " %")
"""
