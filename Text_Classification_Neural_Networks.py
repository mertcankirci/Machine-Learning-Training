from email.utils import decode_params
import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

#Splitting data into two parts

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words= 88000)

#print(train_data) shows integer coded words

word_index = data.get_word_index() #tensorflow has the dictionary for mappings of words , initializing it
word_index = {k:(v+3) for k, v in word_index.items()} #k is the keyword v is the value //integer

#Assigning my own values can do it because we add v+3 in for loop

word_index["<PAD>"] = 0 #padding to make movie reviews same length
word_index["<START>"] = 1
word_index["<UNK>"] = 2 #unknown characters
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) #now we have our dictionary, we set the dictionary as values first keys after that

#Getting all reviews to 250 words by adding padding to them or deleting words

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value= word_index["<PAD>"], padding="post", maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=256)

#Decoding training and testing data to readable words

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text]) #Try to get index i if we cant find the value we'll put ?

#Defining model
"""
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16)) #10000 word vectors and 16 dimensons
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000] #validation data (changing parameters each time untill we get a that is better and more accurate)
x_train = train_data[:10000]

y_val = train_labels[:10000]
y_train = train_labels[:10000]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data = (x_val, y_val), verbose=1)#batch_size means how many reviews we'll load 

results = model.evaluate(test_data, test_labels)

model.save("model.h5") """ #Saving the model  https://www.youtube.com/watch?v=Xmga_snTFBs&list=PLzMcBGfZo4-lak7tiFDec5_ZMItiIIfmj&index=8 test model yourself

keras.models.load_model("model.h5")
