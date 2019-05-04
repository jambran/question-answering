# question-answering
Tackling the NLP task of question answering using SQuAD Dataset (https://rajpurkar.github.io/SQuAD-explorer/). 

Final project for Information Extraction, May 2019.

Inspired by work from Hu, et. al in their paper [Read + Verify: Machine Reading Comprehension
with Unanswerable Questions](https://arxiv.org/pdf/1808.05759.pdf), I provide a model that classifies questions given their context into two categories: possible to answer or impossible. It seems obvious that first predicting the span of the answer then using that span of text to predict whether or not the question can be satisfied (as in Hu, et. al's publication) improves performance. My work sets out to show that is the case. I present a binary classifier trained on the first 10,000 questions from SQuAD. Data trimmed for time limitations.

See experiments in the directory `exp`. 

Find source code in `src`. 

## Requirements
Find all required packages in `requirements.txt`

To install all requirements, use   `pip install -r requirements.txt`

## How to Run
Run from command line using 
```buildoutcfg
python src\train.py --train_file data/train-v2.0.json ^
                    --val_file data/dev-v2.0.json ^
                    --epochs 3 ^
                    --log_dir exp/3epochs/logs ^
                    --exp_dir exp/3epochs ^
                    --data_limit 10000
```

### Download the Data
Visit this [website](https://rajpurkar.github.io/SQuAD-explorer/) to download the data. Put it in the data folder.

### Train
Train calls `load_data.py` first, which preprocesses the dataset. 
1. Parse the original JSON to extract the context, question, and list of possible answers
2. Preprocess the text to use for prediction
    1. split context and question on whitespace and make all words lowercase
    2. pad or trim the question to max length 40
    3. pad or trim the context to max length 300
3. Initialize the model. See architecture in `model.py`
4. Train (warning - takes about 8 hours for 10,000 training samples on my machine)
5. Evaluate. See function in `train.py`

All results are written out the the log file `train.log` located in the folder `log_dir` from the initial call from command line. 
