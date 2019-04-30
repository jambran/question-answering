# coding: utf-8

import io
import json
from Instance import Instance

import gluonnlp as nlp
import mxnet as mx

import utils


def load_json_to_array(data_file):
    """
    load the data into memory
    :param data_file: the file path of the data to load in
    :return: a list of Instance objects
    """
    arr = []
    with open(data_file) as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
        for wiki_article in dataset:
            title = wiki_article['title']
            for paragraph in wiki_article['paragraphs']:
                context = paragraph['context']
                qas = paragraph['qas']
                for qa in qas:
                    i = Instance(answers=qa['answers'],
                                 id=qa['id'],
                                 is_impossible=qa['is_impossible'],
                                 question=qa['question'],
                                 context=context,
                                 title=title,
                                 )
                    arr.append(i)
    return arr


def load_dataset(train_file, val_file, max_length=32, source=None, ctx=mx.cpu()):
    """
    load all data sets into memory
    :param train_file: json format SQuAD data set
    :param val_file:  json format SQuAD data set
    :param max_length: the max padding length for question, context and answer
    :param source: the source for word embeddings
    :param ctx: the CPU or GPU context
    :return:
    """
    train_array = load_json_to_array(train_file)
    val_array = load_json_to_array(val_file)

    vocabulary = build_vocabulary(train_array, val_array, source=source, context=ctx)

    train_dataset = preprocess_dataset(train_array, vocabulary, max_length)
    val_dataset = preprocess_dataset(val_array, vocabulary, max_length)

    return vocabulary, train_dataset, val_dataset


def get_tokens_from(data_array):
    """
    :param data_array: a list of data instances as 4-tuples
    :return: combined list of tokens from each instance in the data set
    """
    all_tokens = []
    for i, instance in enumerate(data_array):
        instance.context = tokenize(instance.context)
        all_tokens.extend(instance.context)

        instance.question = tokenize(instance.question)
        all_tokens.extend(instance.question)

        for answer in instance.answers:
            answer.text = tokenize(answer.text)
            all_tokens.extend(answer.text)
    return all_tokens


def build_vocabulary(tr_array, val_array, source,
                     context=mx.cpu()):
    """
    Inputs: arrays representing the training, validation and test data
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """

    # okay to look at val and tst here because we're only used pre-trained word embeddings
    # want to map these items to an index in vocab so we'll get the pre-trained embedding later
    all_tokens = []
    for arr in [tr_array, val_array]:
        all_tokens.extend(get_tokens_from(arr))
    counter = nlp.data.count_tokens(all_tokens)
    vocab = nlp.Vocab(counter)
    # set pre-trained embeddings
    glove_twitter = nlp.embedding.create('glove', source=source)
    vocab.set_embedding(glove_twitter)
    initialize_random_vectors_for(tokens=['<unk>', '<bos>', '<eos>'], vocab=vocab)
    return vocab


def _preprocess(x, vocab, max_len):
    """
    Inputs: data instance x (tokenized), vocabulary, maximum length of input (in tokens)
    Outputs: data mapped to token IDs, with corresponding label
    """
    x.process_text(vocab, max_len)
    return x.question_indices, x.context_indices, x.answers_indices, [a.start for a in x.answers]
    # label, ind1, ind2 = x
    # return label, ind1, ind2, indices_of_tokens


def preprocess_dataset(dataset, vocab, max_len):
    preprocessed_dataset = [_preprocess(x, vocab, max_len) for x in dataset]
    return preprocessed_dataset


def tokenize(txt):
    """
    Tokenize an input string. Something more sophisticated may help . . .
    """
    return [w.lower() for w in txt.split(' ')]


def initialize_random_vectors_for(vocab, tokens=['<unk>', '<bos>', '<eos>'], mean=0, standard_dev=1):
    """
    randomly initializing weights for unknown words according to standard normal distribution
    :tokens: iterable of tokens to make randomly initialized vectors for
    :mean: when initializing, this is mean for each element in the vector
    :standard_dev:  when initializing, this is the standard deviation for each element in the vector
    :return: None
    """
    shape = vocab.embedding.idx_to_vec[0].shape
    for token in tokens:
        vocab.embedding[token] = mx.nd.random.randn(shape[0], loc=mean, scale=standard_dev)
