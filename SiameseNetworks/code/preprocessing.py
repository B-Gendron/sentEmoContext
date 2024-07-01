# Torch utils
from torch.utils.data import Dataset
import torch
from torchtext.vocab import vocab
torch.set_default_dtype(torch.float32)

# Data loading and preprocessing
import datasets
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict
from nltk.tokenize import TweetTokenizer
from sentence_transformers import SentenceTransformer

# General purposes modules
import numpy as np
import argparse
from transformers import AutoTokenizer

# From another scripts
from utils import *

# ISOLATED UTTERANCE MODEL PREPROCESSING FUNCTIONS

def tokenize_pad_numericalize_dialog(entry, vocab_stoi, max_length=20):
    """
    Performs tokenization and padding at message level.

    Args:
        entry (str): the sentence to process
        vocab_stoi (list): a dict mapping the words to their indexes
        max_length (int): length threshold for padding (default=20)

    Returns:
        padded_dialog (list): the tokenized and padded sentence 
    """
    tok = TweetTokenizer()
    dialog = [ [ vocab_stoi[token] if token in vocab_stoi else 0 for token in tok.tokenize(e.lower()) ] 
            for e in entry ]
    padded_dialog = list()
    for d in dialog:
        if len(d) < max_length:    padded_dialog.append( d + [ 1 for i in range(len(d), max_length) ] )
        elif len(d) > max_length:  padded_dialog.append(d[:max_length])
        else:                      padded_dialog.append(d)
    return padded_dialog


def tokenize_all_dialog(entries, target, vocab_stoi, max_message_length=20, max_dialog_length=12):
    """
    Apply tokenization to the whole dialog. 

    Args:
        entries (list): list of sentences that make up the dialog
        target (str): the type of labels to store, being either 'emotion' or 'act'
        vocab_stoi (list): a dict mapping the words to their indexes
        max_message_length (int): length threshold for padding messages(default=20)
        max_dialog_length (int): length threshold for padding dialogs (default=12)

    Returns:
        res (dict): the tokenized and padded utterances along with the associated labels
    """
    res_dialog, res_labels = [], []

    for entry in entries['dialog']:
        text  = tokenize_pad_numericalize_dialog(entry, vocab_stoi)
        if len(text) < max_dialog_length:    text = text + [ [1] * max_message_length for i in range(len(text), max_dialog_length)]   # pad_message * (max_dialog_length - len(text))
        elif len(text) > max_dialog_length:  text = text[-max_dialog_length:] # keeps the last n messages
        res_dialog.append(text)

    for labels in entries[target]:
        if len(labels) < max_dialog_length:   labels = labels + [ 0 for i in range(len(labels), max_dialog_length) ]          # pad_label * (max_dialog_length - len(labels))
        elif len(labels) > max_dialog_length: labels = labels[-max_dialog_length:]
        res_labels.append(labels)

    res = {'text': res_dialog, 'label': res_labels}
    return res


def apply_tokenization(vocab_stoi):
    '''
        Apply tokenization and padding using the custom functions above.
        Note that this should NOT be used in case you want to train a BERT model. In this case, use the function apply_tokenization_bert.
    '''
    for split in ['train', 'validation', 'test']:
        dailydialog[split] = dailydialog[split].map(lambda e: tokenize_all_dialog(e, 'emotion', vocab_stoi), batched=True)

    return dailydialog


def reshape_data_utterances(dataset, target, max_message_length=20, max_dialog_length=12):
    """
    Reshape the given processed dataset by utterance and by dialog to keep only the information related to the utterances. Please mind that, in order to avoid issues with dimensions, it is assumed here that data is already padded.

    Args:
        dataset (DatasetDict): a huggingface formatted dataset to be reshaped
        target (str): either 'text' (for isolated utterances) or 'embedding' (for contextual utterances using sentence embeddings)
        max_message_length (int): length threshold for padding messages (default=20)
        max_dialog_length (int): length threshold for padding dialogs (default=12)

    Returns:
        x_train: the processed dialogs of the training set
        x_val: the processed dialogs of the validation set
        x_test: the processed dialogs of the test set
        y_train: the processed labels of the training set
        y_val: the processed labels of the validation set
        y_test: the processed labels of the test set
    """
    n_train = dataset['train'].num_rows
    n_val = dataset['validation'].num_rows
    n_test = dataset['test'].num_rows
    x_train = np.array(dataset['train'][target]).reshape((max_dialog_length*n_train, max_message_length))
    x_val = np.array(dataset['validation'][target]).reshape((max_dialog_length*n_val, max_message_length))
    x_test = np.array(dataset['test'][target]).reshape((max_dialog_length*n_test, max_message_length))
    y_train = np.array(dataset['train']['label']).reshape((-1,1))
    y_val = np.array(dataset['validation']['label']).reshape((-1,1))
    y_test = np.array(dataset['test']['label']).reshape((-1,1))
    return x_train, x_val, x_test, y_train, y_val, y_test



def create_dataset_utterances(dataset, target, max_message_length):
    """
    Apply the reshape function to dailydialog data to create a new dataset with an adapted format for utterances.

    Args:
        dataset (DatasetDict): the huggingface formatted dataset to format
        target (str): either 'text' or 'embedding', depending for which experiment this function is used.

    Returns:
        dyda_utterances: correctly formatted dataset
    """
    # set the maximum length for the utterance to the right value according to the model for which data is preprocessed
    dialog_length = 12 if target=='text' else 15

    x_train, x_val, x_test, y_train, y_val, y_test = reshape_data_utterances(dataset, target, max_dialog_length=dialog_length, max_message_length=max_message_length)

    # build the whole dataset
    dyda_utterances = {'train':Dataset.from_dict({'label':y_train,'text':x_train}),
                       'validation':Dataset.from_dict({'label':y_val,'text':x_val}),
                       'test':Dataset.from_dict({'label':y_test,'text':x_test})
        }
    
    return DatasetDict(dyda_utterances)


def create_dataset_utterances_vectors(dataset, target, pretrained_vectors, vocab_itos):
    """
    Map tokens to vectors from FastText in the utterances dataset in order to build the utterances vectors dataset.

    Args: 
        dataset: the isolated utterance dataset
        target (str): either 'text' or 'embedding', depending for which experiment this function is used.

    Returns:
        dyda_utterances_vectors: correctly formatted dataset
    """
    dyda_utterances = create_dataset_utterances(dataset, target, max_message_length=20)

    # apply mapping to all dailydialog entries    
    dyda_utterances_vectors = {'train': dyda_utterances['train'].map(lambda e: map_tokens_to_vectors(e, pretrained_vectors, vocab_itos)), 
                            'validation': dyda_utterances['validation'].map(lambda e: map_tokens_to_vectors(e, pretrained_vectors, vocab_itos)), 
                            'test': dyda_utterances['test'].map(lambda e: map_tokens_to_vectors(e, pretrained_vectors, vocab_itos))}
    
    utterances_vectors = DatasetDict(dyda_utterances_vectors)

    return utterances_vectors


# CONTEXTUAL UTTERANCE MODEL PREPROCESSING FUNCTIONS

def process_dialogs_labels(entries, max_length, sep_token='[SEP]'):
    '''
        Add some right padding to a list of labels.
        
        Args:
            entries: the entries of dailydialog split (train, val, test)
            max_length (int): the maximum length of the dialog, that correspond to the maximun length of the labels list

        Returns:
            labels (list): the padded list of labels.
    '''
    res_dialog, res_labels = [], []

    # process dialogs
    for dialog in entries['dialog']:
        processed_dialog = f"{sep_token}".join(dialog)
        res_dialog.append(processed_dialog)

    # process labels
    for labels in entries['emotion']:
        if len(labels) < max_length:
            labels = labels + [-1 for _ in range(len(labels), max_length)]
        res_labels.append(labels)

    return {"dialog": res_dialog, "emotion": res_labels}

# At this point, the number of labels and utterances is the same
def prepare_data_conversations_bert(max_length=35):
    """
    Organize data by dialog to keep only the information related to the conversation for the rest of the code. This code prepares for BERT tokenization and thus makes use of SEP token.

    Args:
        max_length: the maximum number of utterances to be kept in the dataset conversations. Default if the max number of utterances in DailyDialog dataset.

    Returns:
        x_train: the processed dialogs of the training set
        x_val: the processed dialogs of the validation set
        x_test: the processed dialogs of the test set
        y_train: the processed labels of the training set
        y_val: the processed labels of the validation set
        y_test: the processed labels of the test set
    """
    # process dialogs and labels
    for split in ['train', 'validation', 'test']:
        dailydialog[split] = dailydialog[split].map(lambda e: process_dialogs_labels(e, max_length), batched=True)

    x_train, x_val, x_test = dailydialog['train']['dialog'], dailydialog['validation']['dialog'], dailydialog['test']['dialog']
    # store labels in variables
    y_train = dailydialog['train']['emotion']
    y_val = dailydialog['validation']['emotion']
    y_test = dailydialog['test']['emotion']

    dyda_prepared_conversations = {
        'train':Dataset.from_dict({'label':y_train,'dialog':x_train}),
        'validation':Dataset.from_dict({'label':y_val,'dialog':x_val}),
        'test':Dataset.from_dict({'label':y_test,'dialog':x_test})
        }

    return DatasetDict(dyda_prepared_conversations)


def apply_tokenization_bert(dataset):
    """
    Instantiate BERT tokenizer based on BERT-base and apply tokenization to the processed datasets.

    Args:
        dataset (DatasetDict): the prepared instances of all splits in a DatasetDict huggingface format

    Returns:
        tok_conversations (DatasetDict): the tokenized instances of all splits in a DatasetDict huggingface format
    """
    # define the tokenizer
    def bert_tokenizer(entries):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        return tokenizer(entries["dialog"], padding="max_length", max_length=512, truncation=True)
    
    # apply tokenization to all splits
    dyda_tokenized_conversations = dataset.map(bert_tokenizer, batched=False)

    # some post-processing
    dyda_tokenized_conversations = dyda_tokenized_conversations.remove_columns(["dialog"])
    dyda_tokenized_conversations = dyda_tokenized_conversations.rename_column("label", "labels")
    dyda_tokenized_conversations.set_format("torch")

    tok_conversations = DatasetDict(dyda_tokenized_conversations)

    return tok_conversations


sentence_models = ['all-mpnet-base-v2', 'all-distilroberta-v1', 'all-MiniLM-L12-v2', 'all-MiniLM-L6-v2', 'all-roberta-large-v1']
def apply_sentence_bert(entry, model, max_length, utterance_level=False):
    """
    Apply Sentence-BERT to DailyDialog data. From the produced embedding which is a list of utterance embeddings, we store two variables. The 'encoding' variable joins the utterance embeddings with a manually added [SEP] in order to retrieve the utterance boundaries. The 'embedding' variable consists in the dialog representation in the sense that we flattened Sentence-BERT output.

    Args:
        entry: a DailyDialog entry
        model: a sentence transformers model
        max_length (int): the embedding max length for the selected model
        utterance_level (bool): specifies if the embeddings should be flatten or not, depending on a use as isolated or contextual representations

    Returns:
        result (dict): a dictionary containing the gold labels, the dialogue full text and the associated embeddings computed using S-BERT
    """
    # Padding length = max length = max_dialog_length * max_sentence_bert = 15 * 384 = 5760
    label = entry['emotion']
    text = entry['dialog']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding = model.encode(text, device=device).tolist()
    utterance_limit = 15

    # Pad labels
    if len(label) < utterance_limit:
        label.extend([-1 for _ in range(utterance_limit-len(label))])
    elif len(label) > utterance_limit:
        label = label[:utterance_limit]

    # Pad embedding
    if len(embedding) < utterance_limit:
        embedding = add_sentencebert_random_vectors(embedding, utterance_limit - len(embedding), max_length)
    elif len(embedding) > utterance_limit:
        embedding = embedding[:utterance_limit]

    final_embedding = embedding 
    if not utterance_level:
        final_embedding = custom_flatten(embedding)
        
    result = {'label': label, 'text': text, 'embedding': final_embedding}
    return result


def prepare_data_sentence_bert(dataset, sentence_model, max_length, utterance_level=False):
    """
    A function to wrap up the preprocessing procedure using Sentence-BERT (S-BERT).

    Args:
        dataset:                             the data to preprocess
        sentence_model:                      the sentence transformer model to use
        max_length (int):                    the max utterance length to use in tokenization
        utterance_level (bool):              specifies whether data should be processed as a bag of utterances (using a sentence embedding) or with conservation of the original dialogue structure

    Returns:
        resulting_dataset (DatasetDict):    the processed dataset using S-BERT sentence embeddings
    """
    model = SentenceTransformer(f'sentence-transformers/{sentence_model}')

    for split in ['train', 'validation', 'test']:
        dataset[split] = dataset[split].map(lambda e: apply_sentence_bert(e, model, max_length, utterance_level=utterance_level))

    new_dataset = {
        'train':Dataset.from_dict({
            'label' : dataset['train']['label'],
            'embedding': dataset['train']['embedding']
            }),
        'validation': Dataset.from_dict({
            'label' : dataset['validation']['label'],
            'embedding': dataset['validation']['embedding']
            }),
        'test':Dataset.from_dict({
            'label' : dataset['test']['label'],
            'embedding': dataset['test']['embedding']
            })
        }
    
    resulting_dataset = DatasetDict(new_dataset)
    return resulting_dataset


def format_utt_sentence_dataset(final_dataset, sm):
    """
    This function is used for S-BERT utterance preprocessing. It is particularly used to provide data for the pre-trained emotion classifier, therefore it is intended to be used with data processed as a bag of utterances. It considers the whole processed dataset name as input. It gets rid of neutral utterances and random utterances due to padding.

    Args:
        final_dataset:                       the bag of utterances dataset.
        sm (str):                            name of the sentence BERT model used to process the utterance dataset.
    """
    utt_sentence = final_dataset
    utt_sentence.set_format(type='pandas')

    # convert dataset splits to Pandas dataframes
    df_utt_train = pd.DataFrame(utt_sentence['train'][:])
    df_utt_val = pd.DataFrame(utt_sentence['validation'][:])
    df_utt_test = pd.DataFrame(utt_sentence['test'][:])

    # fetch the label value by creating an auxiliary column
    df_utt_train['lab_val'] = df_utt_train['label'].map(lambda l: l[0])
    df_utt_val['lab_val'] = df_utt_val['label'].map(lambda l: l[0])
    df_utt_test['lab_val'] = df_utt_test['label'].map(lambda l: l[0])

    # remove utterances from padding
    df_utt_train = df_utt_train[df_utt_train['lab_val'].isin([0, 1, 2, 3, 4, 5, 6])]
    df_utt_val = df_utt_val[df_utt_val['lab_val'].isin([0, 1, 2, 3, 4, 5, 6])]
    df_utt_test = df_utt_test[df_utt_test['lab_val'].isin([0, 1, 2, 3, 4, 5, 6])]

    # resample train set (see resample_dataframe() auxialiary function from utils)
    df_utt_train = resample_dataframe(df_utt_train)

    # get rid of the proviously created auxiliary column
    df_utt_train = df_utt_train.drop('lab_val', axis=1)
    df_utt_val = df_utt_val.drop('lab_val', axis=1)
    df_utt_test = df_utt_test.drop('lab_val', axis=1)

    # convert processed Pandas dataframes to DatasetDict object
    train_set = Dataset.from_dict(df_utt_train)
    val_set = Dataset.from_dict(df_utt_val)
    test_set = Dataset.from_dict(df_utt_test)

    final_dataset = DatasetDict({
        'train':train_set,
        'validation':val_set,
        'test':test_set
    })

    save_dataset(final_dataset, f'all_balanced_utterances_{sm}', output_format='huggingface')


if __name__ == "__main__":

    # Instantiate parser - To be improved
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--level", required=True, help="Indicates if the preprocessing should be done at the utterance level ('u'), at the conversation level for BERT ('c'), or at the conversation level for Sentence BERT ('s')", type=str)    
    parser.add_argument("-t", "--type", help="[S-BERT only] Indicates what type of data should be found in the dataset. Either utterances ('u') or on conversations ('c')", type=str)
    arguments = parser.parse_args()

    dailydialog = datasets.load_dataset('daily_dialog') 

    # CONVERSATION-AWARE DIALOG REPRESENTATIONS (SENTENCE-BERT)
    if arguments.level == 's':
        print("Selected preprocessing: ", colored('SENTENCE-BERT conversation-aware dialog representations', 'green'))
        max_lengths = {'all-MiniLM-L6-v2':384, 'all-roberta-large-v1':1024, 'all-mpnet-base-v2':768}
        sentence_models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'all-roberta-large-v1']
        model_count = 1
        
        if arguments.type == 'c':   
            print(colored("Important note: data is processed as conversations", 'green'))
            for sm in sentence_models:
                print(colored(f"Dataset {model_count}/3, Pretrained model:{sm}", 'yellow'))
                prepared_data = prepare_data_sentence_bert(dailydialog, sm, max_lengths[f'{sm}'])
                save_dataset(prepared_data, f"sentences_processed_{sm}", output_format='huggingface')
                model_count += 1

        elif arguments.type == 'u': 
            print(colored("Important note: data is processed as utterances", 'green'))
            for sm in sentence_models:
                print(colored(f"Dataset {model_count}/3, Pretrained model:{sm}", 'yellow'))

                # process data utterance-wise
                prepared_data = prepare_data_sentence_bert(dailydialog, sm, max_lengths[f'{sm}'])
                final_dataset = create_dataset_utterances(prepared_data, 'embedding', max_lengths[f'{sm}'])
                format_utt_sentence_dataset(final_dataset, sm)
                model_count += 1

    # ISOLATED UTTERANCES REPRESENTATIONS
    if arguments.level == 'u':
        print("Selected experiment: ", colored('isolated utterance representations', 'green'))

        # get all the voc and stoi/itos utils for vectors computations
        pretrained_vectors, vocab_itos, vocab_stoi = get_and_set_pretrained_vectors()

        # preform preprocessing
        tokenized_dataset = apply_tokenization(vocab_stoi)
        dyda_utterances_vectors = create_dataset_utterances_vectors(tokenized_dataset, 'text', pretrained_vectors, vocab_itos)
        save_dataset(dyda_utterances_vectors, "utterances_vectors", output_format='huggingface')