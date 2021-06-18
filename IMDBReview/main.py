import tensorflow as tf
from tensorflow import keras

"""tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)"""

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=90000)

word_index = data.get_word_index()
word_index["<PAD>"] = 0

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

"""model = keras.Sequential()

model.add(keras.layers.Embedding(90000,16))
model.add(keras.layers.GlobalAvgPool1D())
model.add(keras.layers.Dense(16,activation="relu"))
model.add(keras.layers.Dense(8,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))

model.summary()

model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitmodel = model.fit(x_train, y_train, epochs=50, batch_size=768, validation_data=(x_val,y_val),verbose=1)

results = model.evaluate(test_data,test_labels)
print(results)

model.save("ImdbModel.h5")"""



def review_encode(s):
	encoded = [1]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded

model = keras.models.load_model("ImdbModel.h5")

with open("test.txt") as f:
    for line in f:
        nline = line.replace(",", "").replace(".","").replace("(","").replace(")","").replace(":","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(predict[0])