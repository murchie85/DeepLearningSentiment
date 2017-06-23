# TFLearn, a high-level library built on top of TensorFlow. TFLearn makes it simpler to build networks just by defining the layers. It takes care of most of the details for you.


import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical


reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)


from collections import Counter
total_counts = Counter()
for _, row in reviews.iterrows():
    total_counts.update(row[0].split(' '))
print("Total words in data set: ", len(total_counts))



#CREATE VOCAB 
#Let's keep the first 10000 most frequent words
vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]
print(vocab[:60])

print(vocab[-1], ': ', total_counts[vocab[-1]])


#CONVERT WORDS TO NUMBERS
word2idx = {word: i for i, word in enumerate(vocab)}







#TEXT TO VECTOR FUNCTION FOR LATER USE
def text_to_vector(text):
    word_vector = np.zeros(len(vocab), dtype=np.int_)			 #empty array of zeroes lenvocab = 10000, we need each one
    for word in text.split(' '):   								  #WORD IS JUST A VARIABLE, THAT EACH ELEMENT IS ASSIGNED TO
        idx = word2idx.get(word, None)   # get the word, apply index function, store as index
        if idx is None:
            continue
        else:
            word_vector[idx] += 1
    return np.array(word_vector)





#Now, run through our entire review data set and convert each review to a word vector.
word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
    word_vectors[ii] = text_to_vector(text[0])


#Train, Validation, Test sets

Y = (labels=='positive').astype(np.int_)
records = len(labels)

shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records*test_fraction)], shuffle[int(records*test_fraction):]
trainX, trainY = word_vectors[train_split,:], to_categorical(Y.values[train_split], 2)
testX, testY = word_vectors[test_split,:], to_categorical(Y.values[test_split], 2)


trainY



# Network building


def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    
    # Inputs
    net = tflearn.input_data([None, 10000]) #input vectors are 10,000 long

    # Hidden layer(s)
    net = tflearn.fully_connected(net, 200, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')

    # Output layer
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', 
                             learning_rate=0.1, 
                             loss='categorical_crossentropy')
    
    model = tflearn.DNN(net)
    return model



#Intializing the model

model = build_model()

# Training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=100)


predictions = (np.array(model.predict(testX))[:,0] >= 0.5).astype(np.int_)
test_accuracy = np.mean(predictions == testY[:,0], axis=0)
print("Test accuracy: ", test_accuracy)






