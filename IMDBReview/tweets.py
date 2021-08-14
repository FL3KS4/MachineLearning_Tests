from tensorflow import keras
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer

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
test_neg3 = test_neg3.sample(15000)

test_pos3 = data[data['target'] == 1]
test_pos3 = test_pos3.sample(15000)


dataset3 = pd.concat([test_pos3, test_neg3])
dataset3 = shuffle(dataset3)


"""Tokenizer"""
myTokenizer = Tokenizer(num_words=120000)
myTokenizer.fit_on_texts(dataset.iloc[:,1])


train_data = myTokenizer.texts_to_sequences(dataset.iloc[:,1])
test_data = myTokenizer.texts_to_sequences(dataset3.iloc[:,1])

word_index = myTokenizer.word_index
word_index["<PAD>"] = 0


train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=150)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=150)

"""test_neg2 = data[data['target'] == 0]
test_neg2 = test_neg2.sample(10000)

test_pos2 = data[data['target'] == 1]
test_pos2 = test_pos2.sample(10000)


dataset2 = pd.concat([test_pos2, test_neg2])
dataset2 = shuffle(dataset2)"""


"""Training and validating"""


"""x_val = train_data[:10000]
y_val = dataset.iloc[:,0][:10000]"""

"""x_train = train_data
y_train = dataset.iloc[:,0]

model = keras.Sequential()

model.add(keras.layers.Embedding(120000,16))
model.add(keras.layers.GlobalAvgPool1D())
model.add(keras.layers.Dense(16,activation="relu"))
model.add(keras.layers.Dense(8,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))

model.summary()

model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])


fitmodel = model.fit(x_train, y_train, epochs=25, batch_size=768, verbose=1)

results = model.evaluate(test_data,dataset3.iloc[:,0])
print(results)

model.save("TweetsModel.h5")"""

def review_encode(s):
	encoded = [1]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded

model = keras.models.load_model("TweetsModel.h5")

with open("test.txt") as f:
    for line in f:
        nline = line.replace(",", "").replace(".","").replace("(","").replace(")","").replace(":","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        result = predict[0] * 100
        print(str(result) + " %")
