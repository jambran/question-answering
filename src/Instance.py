import mxnet as mx


def process_text(words, vocab, max_len):
    indices = vocab[words]  ## map tokens (strings) to unique IDs
    indices = indices[:max_len]  ## truncate to max_len
    # pad if necessary
    while len(indices) < max_len:
        indices.append(vocab['<pad>'])
    assert len(indices) == max_len
    return mx.nd.array(indices)


class Instance:
    def __init__(self, answers: list,
                 id: str,
                 is_impossible: bool,
                 question: str,
                 context: str,
                 title: str):
        """

        :param answers: list of dictionaries with keys 'answer_start' and 'text'
        :param id: unique identifier
        :param is_impossible: can the question be answered from the context
        :param question: the question posed by a crowd-source worker
        :param context: a partial of the wikipedia article that may answer the question
        :param title: title of the wikipedia article
        """
        self.answers = [Answer(start=answer['answer_start'],
                               text=answer['text'])
                        for answer in answers]
        self.id = id
        self.is_impossible = is_impossible
        self.question = question
        self.context = context
        self.title = title

    def process_text(self, vocab, max_len):
        """
        get lists of the indices of the vocab items
        :param vocab: the mapping of vocab item to integer
        :param max_len: the max padding length
        :return: None
        """
        self.question_indices = process_text(self.question, vocab, max_len)
        self.context_indices = process_text(self.context, vocab, max_len)


class Answer:
    def __init__(self,
                 start: int,
                 text: str):
        """

        :param start: the span start in the context to find the answer
        :param text: the words that answer the question
        """
        self.start = start
        self.text = text
