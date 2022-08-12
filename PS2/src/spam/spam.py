import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE *** 
    mess = message.lower()
    words = mess.split(' ')
    return words
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    # result = []
    # for mess in messages: 
    #     words = set(get_words(mess))
    #     # here a problem! Fix! check every message! 
    #     result += get_words(mess)
    # word_list = [item for item, count in collections.Counter(result).items() if count > 4]
    
    # vocab = dict()
    # for i, word in enumerate(word_list):
    #     vocab[word] = i 
    vocab = dict()
    # words = [i for j in messages for i in get_words(j)]
    for i in messages:
        words = set(get_words(i))
        for word in words:
            if word in vocab.keys():
                vocab[word] += 1
            else:
                vocab[word] = 1 
    vocab = vocab(filter(lambda elem: elem[1] > 4, vocab.items()))
    return {w:i for i,w in enumerate(vocab.keys())}

   
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    words = np.zeros((len(messages),len(word_dictionary)))

    for i, msg in enumerate(messages): 
        word_list = get_words(msg)
        counter = collections.Counter(word_list)
        for item, count in counter.items():
            if item in word_dictionary: 
                words[i, word_dictionary[item]] = count
    return words 
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    size_vocab = matrix.shape[1]
    phi = np.mean(labels) 
    cls0 = matrix[labels == 0]
    cls1 = matrix[labels == 1]

    #setting parameters for class 0 
    phi0 = np.sum(cls0, axis=0)
    phi0 = (phi0 + 1.) / (phi0 + size_vocab)

    #setting parameters for class 1
    phi1 = np.sum(cls1, axis=0)
    phi1 = (phi1 + 1.) / (phi1 + size_vocab)
    return phi, phi0, phi1

   
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    phi, phi0, phi1 = model
    mess0 = matrix * np.log(phi0)

    log_cls0 = np.log(1 - phi) + np.sum(mess0, axis=1)  
    mess1 = matrix * np.log(phi1)

    log_cls1 = np.log(phi) + np.sum(mess1, axis=1) 
    pred = (log_cls1 > log_cls0).astype(np.int64)
    return pred
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    phi, phi0, phi1 = model
    log_diff = np.log(phi1) - np.log(phi0)

    top_wd = np.argsort(log_diff)
    wd_idx = {i: word for word, i in dictionary.items()}
    top_wd = [wd_idx[idx] for idx in top_wd[-5:]]
    return top_wd
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    best_rad = None
    best_acc = 0
    for r in radius_to_consider:
        test_pred = svm.train_and_predict_svm(train_matrix, train_labels, 
                                                val_matrix, r)
        test_acc = np.mean(test_pred == val_labels)
        if test_acc > best_acc:
            best_rad = r
            best_acc = test_acc
    return best_rad
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
