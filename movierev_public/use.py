import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
word_tokenizer = RegexpTokenizer(r'\w+')

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 5

batch_size = 32
total_batches = int(128000/batch_size)
hm_epochs = 10

lexicon_length = 5068

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([lexicon_length, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']
    return output
    
saver = tf.train.Saver()

def use_neural_network(input_data):
    prediction = neural_network_model(x)
    with open('lexicon.pickle','rb') as f:
        lexicon = pickle.load(f)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,"./model.ckpt")
        current_words = word_tokenizer.tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        print("\n")
        print("Recognized words:")
        for word in current_words:
            if word.lower() in lexicon:
                print(" - " + str(word))
                index_value = lexicon.index(word.lower())
                features[index_value] += 1

        features = np.array(list(features))

        result = sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1))

        if(result[0] == 0):
            print("1 star ::: " + input_data)
        else:
            print(str(result[0]+1) + " stars ::: " + input_data)
        
        print("\n")
        return result[0]
    
use_neural_network("The director failed in every possible way. Worst movie ever.")
while(True):
    use_neural_network(input("Enter a review: "))
