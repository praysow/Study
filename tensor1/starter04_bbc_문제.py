# # Question
# #
# # For this task you will train a classifier on the bbc news text archive
# # Your classifier should be trained on the 6 categories shown below, and your final layer
# # should be as shown -- Dense layer with 6 neurons and softmax activation.
# # Your model will be tested against unseen sentences and you will be scored on whether they
# # are correctly classified or not.


# import csv
# import tensorflow as tf
# import numpy as np
# import urllib
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences


# def solution_model():
#     url = 'https://storage.googleapis.com/download.tensorflow.org/data/bbc-text.csv'
#     urllib.request.urlretrieve(url, 'bbc-text.csv')

#     # DO NOT CHANGE THE VALUES OF THESE CONSTANTS OR THE TESTS MAY FAIL
#     vocab_size = 1000
#     embedding_dim = 16
#     max_length = 120
#     trunc_type='post'
#     padding_type='post'
#     oov_tok = "<OOV>"
#     training_size = 20000
#     sentences = []
#     labels = []
#     stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


#     with open("bbc-text.csv", 'r') as csvfile:
#         # YOUR CODE HERE

#     model = tf.keras.Sequential([
#     # YOUR CODE HERE
#     # PLEASE NOTE -- WHILE THERE ARE 5 CATEGORIES, THEY ARE NUMBERED 1 THROUGH 5 IN THE DATASET
#     # SO IF YOU ONE-HOT ENCODE THEM, THEY WILL END UP WITH 6 VALUES, SO THE OUTPUT LAYER HERE
#     # SHOULD ALWAYS HAVE 6 NEURONS AS BELOW. MAKE SURE WHEN YOU ENCODE YOUR LABELS THAT YOU USE
#     # THE SAME FORMAT, OR THE TESTS WILL FAIL
#     # 0 = UNUSED
#     # 1 = SPORT
#     # 2 = BUSINESS
#     # 3 = POLITICS
#     # 4 = TECH
#     # 5 = ENTERTAINMENT
#         tf.keras.layers.Dense(6, activation='softmax')
#     ])

#     # YOUR CODE HERE
#     return model


# # Note that you'll need to save your model as a .h5 like this
# # This .h5 will be uploaded to the testing infrastructure
# # and a score will be returned to you
# if __name__ == '__main__':
#     model = solution_model()
#     model.save("mymodel.h5")

import csv
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/bbc-text.csv'
    urllib.request.urlretrieve(url, 'bbc-text.csv')

    # DO NOT CHANGE THE VALUES OF THESE CONSTANTS OR THE TESTS MAY FAIL
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000
    sentences = []
    labels = []

    # Load the data from the CSV file
    with open("bbc-text.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) # Skip header
        for row in reader:
            labels.append(row[0])  # Get the label
            sentences.append(row[1])

    # Create a dictionary to map labels to integers
    label_to_index = {label: index for index, label in enumerate(set(labels))}

    # Convert labels to integers using the dictionary
    labels = [label_to_index[label] for label in labels]

    # Preprocess the data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Convert labels to one-hot encoding
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_to_index))

    # Split data into training and validation sets
    training_sentences = padded_sequences[:training_size]
    training_labels = labels[:training_size]
    validation_sentences = padded_sequences[training_size:]
    validation_labels = labels[training_size:]

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(len(label_to_index), activation='softmax')  # Use len(label_to_index) as the number of output classes
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(training_sentences, training_labels, epochs=10, validation_data=(validation_sentences, validation_labels))

    # Print the accuracy at each epoch
    print("Training accuracy:")
    print(history.history['accuracy'])
    # print("Validation accuracy:")
    # print(history.history['val_accuracy'])

    return model

if __name__ == '__main__':
    model = solution_model()
    # model.save("mymodel.h5")
'''
[0.24853932857513428, 0.45617976784706116, 0.6516854166984558, 0.7424719333648682, 0.8359550833702087, 0.9137078523635864, 0.9258427023887634, 0.9415730237960815, 0.951910138130188, 0.9568539261817932]
'''