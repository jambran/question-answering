# codeing: utf-8

import argparse
import logging
import os
import pickle

import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet import autograd
import gluonnlp as nlp

from load_data import load_dataset
from model import QuestionAnsweringClassifier
import utils

LOSS_FN = gluon.loss.SoftmaxCrossEntropyLoss()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a (short) text classifier - via attention based architecture')
    parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data')
    parser.add_argument('--val_file', type=str, help='File containing file representing the input VALIDATION data',
                        default=None)
    parser.add_argument('--epochs', type=int, default=4, help='Upper epoch limit')
    parser.add_argument('--optimizer', type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.0001)
    parser.add_argument('--batch_size', type=int, help='Training batch size', default=16)
    parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.5)
    parser.add_argument('--embedding_source', type=str, default='glove.twitter.27B.100d',
                        help='Pre-trained embedding source name')
    parser.add_argument('--log_dir', type=str, default=os.path.join('..', 'exp', 'logs'),
                        help='Output directory for log file')
    parser.add_argument('--exp_dir', type=str, default=os.path.join('..', 'exp', 'debug'),
                        help='Directory to hold all files pertaining to this experiment')
    parser.add_argument('--fixed_embedding', action='store_true', help='Fix the embedding layer weights')
    parser.add_argument('--random_embedding', action='store_true', help='Use random initialized embedding layer')
    parser.add_argument('--attn_cell', type=str, help='the type of attention cell to use', default='multi_head')

    return parser.parse_args()


def make_dataloader(data, shuffle):
    """
    given a dataset, return a dataloader
    :param data: the dataset to create a dataloader for
    :param transformer:
    :param shuffle: boolean whether or not the DataLoader should be shuffled
    :return: dataloader
    """
    data = gluon.data.SimpleDataset(data)
    dataloader = mx.gluon.data.DataLoader(data, batch_size=ARGS.batch_size, shuffle=shuffle)
    return dataloader


def train_classifier(vocab, data_train, data_val, ctx=mx.cpu()):
    """

    :param vocab: vocab object with embeddings attached
    :param transformer: the BasicTransform object with the map from label to index
    :param data_train: the training data
    :param data_val: the validation data
    :param ctx: the context to use
    :return: the model that was trained
    """
    train_dataloader = make_dataloader(data_train, shuffle=True)
    val_dataloader = make_dataloader(data_val, shuffle=True)

    emb_input_dim, emb_output_dim = vocab.embedding.idx_to_vec.shape

    num_classes = 2
    model = QuestionAnsweringClassifier(emb_input_dim,
                                        emb_output_dim,
                                        num_classes=num_classes,
                                        dropout=ARGS.dropout,
                                        attn_cell=ARGS.attn_cell)

    ## initialize model parameters on the context ctx
    model.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx, force_reinit=True)

    if not ARGS.random_embedding:
        # set the embedding layer parameters to pre-trained embedding
        model.embedding.weight.set_data(vocab.embedding.idx_to_vec)
    elif ARGS.fixed_embedding:
        # don't let the model update the embeddings created when training
        model.embedding.collect_params().setattr('grad_req', 'null')

    # model.hybridize() ## OPTIONAL for efficiency - perhaps easier to comment this out during debugging

    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': ARGS.lr})

    formatting_header, formatting_log = utils.format_epoch_updates()
    logger.info(formatting_header % ('Epoch', 'Train loss', 'Train Acc', 'Val Acc'))

    for epoch in range(ARGS.epochs):
        epoch_loss = 0
        for i, instance in enumerate(train_dataloader):
            question_indices, context_indices, can_be_answered = instance
            question_indices = question_indices.as_in_context(ctx)
            context_indices = context_indices.as_in_context(ctx)
            can_be_answered = can_be_answered.as_in_context(ctx)
            with autograd.record():
                # output shape should (batch_size, num_classes, maybe1?) maybe 2 dimensional
                output = model(question_indices, context_indices)
                l = LOSS_FN(output, can_be_answered).mean()
            l.backward()
            trainer.step(1)  ## step based on batch size
            epoch_loss += l.asscalar()

        val_acc = evaluate(model, val_dataloader)
        train_acc = evaluate(model, train_dataloader)
        logger.info(formatting_log % (epoch, epoch_loss, train_acc, val_acc))

    return model


def evaluate(model, dataloader, ctx=mx.cpu()):
    """
    Get predictions on the dataloader items from model
    Return metrics (accuracy, etc.)
    """
    acc = 0
    total_correct = 0
    total = 0
    all_scores = []
    all_labels = []
    for i, (data, inds, label) in enumerate(dataloader):
        out = model(data, inds)
        predictions = mx.nd.argmax(out, axis=1).astype('int32')
        for j in range(out.shape[0]):
            probs = mx.nd.softmax(out[j]).asnumpy()
            lab = int(label[j].asscalar())
            best_probs = np.argmax(probs)
            if lab == best_probs:
                total_correct += 1
            total += 1
    acc = total_correct / float(total)
    return acc


def predict_on(dataloader, outfile, transform):
    with open(outfile, 'w+') as f:
        for i, (data, inds, label) in enumerate(dataloader):
            out = model(data, inds)
            predictions = mx.nd.argmax(out, axis=1).astype('int32')
            labels = [transform._labels[pred] for pred in predictions.asnumpy()]
            f.write('\n'.join(labels) + '\n')


if __name__ == '__main__':
    ARGS = parse_args()
    logger = logging.getLogger('train.py')
    utils.logging_config(folder=ARGS.log_dir,
                         name='train',
                         level=logging.INFO,
                         console_level=logging.INFO)

    logger.info("Beginning experiment:")
    utils.log_experiment_parameters(ARGS, logger)

    # ensure experiment directory has been created
    if not os.path.exists(ARGS.exp_dir):
        os.makedirs(ARGS.exp_dir)

    ctx = utils.check_for_gpu(logger=logger)

    logger.info("Loading in the dataset...")
    vocab, train_dataset, val_dataset = load_dataset(train_file=ARGS.train_file,
                                                     val_file=ARGS.val_file,
                                                     source=ARGS.embedding_source)
    logger.info("Data loaded in successfully")

    logger.info(f"Training the classifier using {ARGS.train_file}...")
    model = train_classifier(vocab, train_dataset, val_dataset, ctx)
    logger.info("Training completed successfully")

    logger.info(f"Saving model..")
    model_file = os.path.join(ARGS.exp_dir, 'model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved here: {model_file}")
