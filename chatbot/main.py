import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", 'rb') as f:
        #save all these variables to a file
        words,labels,training, output = pickle.load(f)
except:
    #data is a dicttionary with the values of json
    words = []
    labels = []
    docs_x = []
    docs_y = []

    #takes all dictionarys
    for intent in data["intents"]:
        #takes all patern words/sentences
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            #wrds converts sentences into list of words
            words.extend(wrds) #add all words in words cause append takes all list
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])


    #stemming = takes words and returns the root word
    words = [stemmer.stem(w.lower()) for w in words if w not in ["?", ",", "."]]
    #remove dublicates
    words = sorted(list(set(words)))

    labels = sorted(labels)

    #bag of words = like 1-hot. Determines the frequence of words contained.

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    #enumarate contains index and value
    for x, doc in enumerate(docs_x):
        bag = []
        #stem words that were contained in docs.(previously we stemmed the ones on words)
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1) #if thew word exists

            else:
                bag.append(0)

        output_row = out_empty[:]
        #place 1 in the vector where the category is contained
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    #change lists to array in order to feed them into model
    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", 'wb') as f:
        #save all these variables to a file
        pickle.dump((words, labels, training, output), f)


from tensorflow.python.framework import ops
#get rid of previous settings
ops.reset_default_graph()

#model expets to have array of length as training
net = tflearn.input_data(shape = [None, len(training[0])])
#add this fully conected layer to our neural network
net = tflearn.fully_connected(net,8)#hiddene layers
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation = "softmax") #output layer
#softmax activation gives output as probabilities to belong to a class
net = tflearn.regression(net)

model = tflearn.DNN(net)


#delete try except and rerun for training new data
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


#delete the files and rerun to make changes

#--------------------------------------------------%%-----------------------------------------------------
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se: #current word is in sentence
                bag[i] = 1

    return np.array(bag)

def chat():
    print("Start talking with the bot(type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words(inp, words)])[0]

        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg["responses"]

            print(random.choice(responses))
        else:
            print("I do not understand try again.")

chat()