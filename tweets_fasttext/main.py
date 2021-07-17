import csv
import pandas as pd
import fasttext

dataset_columns = ['target', 'id', 'date', 'flag', 'user', 'text']
dataset_encoding = "ISO-8859-1"
df = pd.read_csv('data/training.csv', encoding=dataset_encoding, names=dataset_columns)

data = df[['target', 'text']]

data['target'] = data['target'].replace(4, 1)

data_neg = data[data['target'] == 0]
data_neg = data_neg.sample(750000)
# data_pos.head()

data_pos = data[data['target'] == 1]
data_pos = data_pos.sample(750000)
# data_pos.head()

data_neg.iloc[:, 0] = data_neg.iloc[:, 0].apply(lambda x: '__label__' + str(x))
data_pos.iloc[:, 0] = data_pos.iloc[:, 0].apply(lambda x: '__label__' + str(x))

dataset = pd.concat([data_pos, data_neg])
print(dataset.head())

data_neg2 = data[data['target'] == 0]
data_neg2 = data_neg2.sample(15000)
# data_pos.head()

data_pos2 = data[data['target'] == 1]
data_pos2 = data_pos2.sample(15000)
# data_pos.head()

data_neg2.iloc[:, 0] = data_neg2.iloc[:, 0].apply(lambda x: '__label__' + str(x))
data_pos2.iloc[:, 0] = data_pos2.iloc[:, 0].apply(lambda x: '__label__' + str(x))

dataset2 = pd.concat([data_pos2, data_neg2])




dataset[['target', 'text']].to_csv('train.txt',
                                          index = False,
                                          sep = ' ',
                                          header = None,
                                          quoting = csv.QUOTE_NONE,
                                          quotechar = "",
                                          escapechar = " ")

dataset2[['target', 'text']].to_csv('test.txt',
                                     index = False,
                                     sep = ' ',
                                     header = None,
                                     quoting = csv.QUOTE_NONE,
                                     quotechar = "",
                                     escapechar = " ")

model = fasttext.train_supervised('train.txt',epoch=25, wordNgrams=3, lr=1.0)

# Evaluating performance on the entire test file
print(model.test('test.txt'))


print(dataset2.iloc[2,1])
print(model.predict(dataset2.iloc[2, 1]))

#model.save_model('fasttext_model')


