import os
import json
from termcolor import colored
import torch
import random
from math import sqrt
from datetime import datetime
from datasets import Dataset, DatasetDict
from collections import Counter
import datasets
import random
from sklearn.utils import resample
import pandas as pd
from torchtext.vocab import vocab, FastText


class EarlyStopping:
    """
    An implementation of an early stopping strategy.

    This class monitors the difference between training and validation losses and triggers early stopping
    if the difference does not improve beyond a specified minimum delta for a given tolerance threshold.

    Inspired from: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

    Args:
        tolerance (int): The number of epochs to wait after validation loss stops improving before stopping training.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.

    Attributes:
        tolerance (int): The number of epochs to tolerate without improvement in validation loss.
        min_delta (float): Minimum change required to qualify as an improvement.
        counter (int): Counter to keep track of epochs without improvement.
        early_stop (bool): Flag to indicate whether to stop training early.

    Methods:
        __call__(train_loss, validation_loss):
            Checks if validation loss has stopped improving, updates counter, and sets early_stop flag accordingly.
    """
    def __init__(self, tolerance=5, min_delta=0):
        """
        Initialize EarlyStopping with tolerance and min_delta.

        Args:
            tolerance (int): The number of epochs to wait after validation loss stops improving before stopping training.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        """
        Check if validation loss has stopped improving and update early_stop flag.

        Args:
            train_loss (float): Current training loss.
            validation_loss (float): Current validation loss.
        """
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    Samples elements randomly from a given list of indices for an imbalanced dataset.

    Args:
        dataset (Dataset): The dataset from which to sample.
        level (str): Specifies whether to sample at 'dialog' or 'utterance' level (default is 'dialog').
        indices (list, optional): A list of indices to sample from. Defaults to None.
        num_samples (int, optional): Number of samples to draw. Defaults to None.
        spreading (bool, optional): Whether to apply label spreading technique. Defaults to False.

    Attributes:
        indices (list): List of indices to sample from.
        num_samples (int): Number of samples to draw.
        dataset (Dataset): The dataset from which to sample.
        spreading (bool): Flag indicating whether to apply label spreading technique.
        level (str): Specifies the sampling level ('dialog' or 'utterance').

    Methods:
        __iter__():
            Generates indices to sample from based on the specified level and spreading technique.
        __len__():
            Returns the length of the dataset, indicating the number of samples to draw.
    """

    def __init__(self, dataset, level='dialog', indices=None, num_samples=None, spreading=False):
        """
        Initialize ImbalancedDatasetSampler with dataset, sampling level, indices, number of samples, and spreading flag.

        Args:
            dataset (Dataset): The dataset from which to sample.
            level (str): Specifies whether to sample at 'dialog' or 'utterance' level (default is 'dialog').
            indices (list, optional): A list of indices to sample from. Defaults to None.
            num_samples (int, optional): Number of samples to draw. Defaults to None.
            spreading (bool, optional): Whether to apply label spreading technique. Defaults to False.
        """
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = num_samples
        self.dataset = dataset
        self.spreading = spreading
        self.level = level

        if self.level == 'dialog':
            with open('../data/all_emotions_dd_train.txt', 'r') as fe:
                emotions = [eval(emotions) for emotions in fe]
                self.emotions_without_pad = emotions[:emotions.tolist().index(-1)] if emotions[len(emotions)-1]==-1 else emotions

    def __iter__(self):
        """
        Generate indices to sample from based on the specified level and spreading technique.

        Yields:
            int: Indices to sample from.
        """
        if self.level == 'dialog':
            if not self.spreading:
                balanced_list = sample_indexes_from_dialogues(self.emotions_without_pad)
                yield from iter(balanced_list)
            else:
                self.emotions_without_pad = [ emotional_label_spreading(emotion) for emotion in self.emotions_without_pad ]
                balanced_list = sample_indexes_from_dialogues_spread(self.emotions_without_pad)
                yield from iter(balanced_list)

        else:
            emotions = [self.dataset[i]['label'] for i in range(len(self.dataset))]
            balanced_list = sample_indexes_from_utterances(emotions)
            yield from iter(balanced_list)

    def __len__(self):
        """
        Returns the length of the dataset, indicating the number of samples to draw.

        Returns:
            int: Length of the dataset.
        """
        return len(self.dataset) # use as many generated indexes as there are elements in the original dataset (provided the dataset class considers the entire dataset)


def save_all_train_set_emotions():
    '''
        Store all the emotions from DailyDialog train set in a txt file to avoid memory issues when performing random sampling.
    '''
    dailydialog = datasets.load_dataset('daily_dialog')
    trainset = dailydialog['train']
    for idx in range(len(trainset)):
        with open('../data/all_emotions_dd_train.txt', 'a') as f:
            print(trainset[idx]['emotion'], file=f)


def sample_indexes_from_utterances(emotions):
    """
    Sample indexes from utterances based on given emotion labels.

    This function generates a list of indexes corresponding to selected utterances
    based on the provided emotion labels. The selection criteria vary depending on
    the emotion label using a probability distribution to sample each label according
    to its frequency in the dataset.

    Args:
        emotions (list): A list of integers representing emotion labels for each utterance.

    Returns:
        selected_indexes (list): A list of selected indexes based on the provided emotion labels.
    """
    selected_indexes = []
    # coeffs = {1:avg/816, 2:avg/297, 3:avg/146, 4:avg/10822, 5:avg/952, 6:avg/1580}
    idx = 0
    for emo in emotions:
        p = random.random()

        if emo == 3:
            selected_indexes.extend([idx for _ in range(17)])
        elif emo == 2:
            selected_indexes.extend([idx for _ in range(8)])
        elif emo == 1:
            selected_indexes.extend([idx for _ in range(5)])
        elif emo == 5:
            selected_indexes.extend([idx for _ in range(2)])
        elif emo == 6:
            selected_indexes.append(idx)
        elif emo == 4 and p > 0.8:
            selected_indexes.append(idx)

        idx += 1

    return selected_indexes

def sample_indexes_from_dialogues(emotions_without_pad):
    """
    Sample indexes from dialogues based on emotion labels to balance the dataset.

    This function is used within a random sampler to select dialogue indexes from the dataset.
    It aims to balance the output dataset with respect to emotional classes by adjusting the
    probabilities to penalize the most frequent classes.

    Args:
        emotions_without_pad (list): A list of emotion labels from DailyDialog dataset.

    Returns:
        selected_indexes (list): A list of selected indexes based on the provided emotion labels.
    """
    selected_indexes = []
    flag = ''
    for _ in range(5):
        idx = 0
        for emotions in emotions_without_pad:
            if sum(emotions) == 0:
                flag = 'neutral'
            elif 4 in emotions and all([1, 2, 3, 5, 6]) not in emotions:  
                flag = 'happiness'
            else:
                flag = 'other'
                
            p = random.random()
            if flag == 'neutral' and p > 0.99: 
                selected_indexes.append(idx)
            elif flag == 'happiness' and p > 0.83: # 5*1/6 = 0.83 (computed from probabilities of other emotional classes)
                selected_indexes.append(idx)
            elif flag == 'other' and p > 0.14: # 1/7 
                selected_indexes.append(idx)

            idx += 1

    return selected_indexes


def sample_indexes_from_dialogues_spread(emotions_without_pad):
    """
    Sample indexes from dialogues based on spreaded emotion labels to balance the dataset.

    This function is used within a random sampler to select dialogue indexes from the dataset.
    It aims to balance the output dataset with respect to emotional classes, where the input
    consists of spreaded labels (lists containing identical values).

    Args:
        emotions_without_pad (list): A list of spreaded emotion labels from DailyDialog dataset.

    Returns:
        selected_indexes (list): A list of selected indexes based on the provided spreaded emotion labels.
    """
    selected_indexes = []
    represented_emotions = [ emotions_without_pad[i][0] for i in range(len(emotions_without_pad)) ]
    flags = {0: 'neutral', 1:'other', 2:'other', 3:'other', 4:'happiness', 5:'other', 6:'other'}
    # for _ in range(2):
    idx = 0
    for emo in represented_emotions:
        p = random.random()

        if emo == 3:
            selected_indexes.extend([idx for _ in range(15)])
        elif emo == 2:
            selected_indexes.extend([idx for _ in range(7)])
        elif emo == 1:
            selected_indexes.extend([idx for _ in range(4)])
        elif emo == 5:
            selected_indexes.extend([idx for _ in range(4)])
        elif emo == 6:
            selected_indexes.append(idx)
        elif emo == 4 and p > 0.8:
            selected_indexes.extend([idx for _ in range(4)])
        elif emo == 0 and p > 0.9:
            selected_indexes.extend([idx for _ in range(2)])

        idx += 1

    return selected_indexes


def emotional_label_spreading(emotions):
    """
    Perform label spreading on a list of emotions to address imbalance issues.

    This function propagates a relatively rare emotional label that is already present 
    in the list to all elements of the list, aiming to balance the emotional distribution.

    Args:
        emotions (list): A one-dimensional list of emotions representing a single DailyDialog data sample.

    Returns:
        list: A list of emotions corresponding to the same sample after applying emotional label spreading.
    """
    emo_label = 0
    n = len(emotions)
    n_without_pad = emotions.index(-1) if -1 in emotions else n
    padded_part = emotions[n_without_pad:]
    i = 0
    while i < n_without_pad:
        emo = emotions[i]

        # if we reach neutrality, we keep searching
        if emo == 0:
            i += 1

        # if we reach "happiness" label, we store this information and keep searching
        elif emo == 4:
            i += 1
            emo_label = emo

        # if we reach a rare emotional label, we are done
        else:
            return [emo for _ in range(n_without_pad)] + padded_part  
    
    # if this point is reached, nothing has been returned, meaning no rarer label than either 0 or 4 has been found
    return [emo_label for _ in range(n_without_pad)] + padded_part


def custom_flatten(ll):
    """
    Flatten a list of lists where sublists are of heterogeneous sizes.

    Args:
        ll (list): A list of lists where sublists can have different sizes.

    Returns:
        l (list): The flattened list containing all elements from the input list of lists.
    """
    l = []
    for sl in ll:
        l.extend(sl)
    return l


def get_datetime():
    """
    Get the current date and time and return it as a formatted string.

    Returns:
        str: Current date and time formatted as a string.
    """
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    return dt_string


def args2filename(dico):
    """
    Build a file name based on the parameters of the experiment given in a dictionary.

    Args:
        dico (dict): A dictionary containing parameters for the experiment.

    Returns:
        filename (str): A string to name a file based on the parameters' nature and values.
    """
    filename = "_".join([f"{k}{v}" for k,v in dico.items()])
    return filename


def map_index_to_emotions(i):
    """
    Map DailyDialog integer labels to the corresponding emotion as a string.

    Args:
        i (int): The DailyDialog label index.

    Returns:
        str: The corresponding emotion.
    """
    emotions = {'0':"no emo.", '1':"anger", '2':"disgust", '3':"fear", '4':"happiness", '5':"sadness", '6':"surprise"}
    return emotions[str(i)]


def load_dataset(dataset_name):
    """
    Load a dataset saved in the HuggingFace format, located in the data folder.

    Args:
        dataset_name (str): The name of the dataset, corresponding to the associated folder inside the data folder.

    Returns:
        dataset (DatasetDict): The dataset as a DatasetDict object.

    Raises:
        Exception: If no folder with the given name is found in the data folder.
    """
    path = f"../data/{dataset_name}"
    if os.path.isdir(path):
        dataset = DatasetDict.load_from_disk(f"../data/{dataset_name}")
    else: 
        raise Exception("No folder with the such name found in the data folder.")

    return dataset


def save_dataset(dataset, dataset_name, output_format='huggingface'):
    """
    Save the dataset into a specified format (HuggingFace or JSON) using the save_to_disk method.

    Args:
        dataset (DatasetDict): The dataset to save in a DatasetDict format.
        dataset_name (str): The name to give to the saved file.
        output_format (str): The format to save the dataset in. Options are 'huggingface' or 'json'. Default is 'huggingface'.

    Raises:
        ValueError: If the given output format is not recognized ('huggingface' or 'json').
    """
    if output_format == "huggingface":
        dataset.save_to_disk(f'../data/{dataset_name}')
    elif output_format == "json":
        with open(f"../data/{dataset_name}.json", 'w') as f:
            json.dump(dataset, f)
    else:
        print("The given output format is not recognized. Please note that accepted formats are 'huggingface' and 'json")


def get_labels_weights(training_set, device, num_labels=7, index=0, penalty=1e9, apply_penalty=True):
    """
    Compute class weights based on the label distribution in the training set, with an optional penalty on a specific label.

    Args:
        training_set (DatasetDict): The training samples to compute the weights on. This can be the entire training set or just a batch.
        device: The device on which computations should be performed.
        num_labels (int): The number of labels for multiclass classification. Default is 7, the number of emotion labels in DailyDialog.
        index (int): The index of the label to further penalize.
        penalty (float): The magnitude of the penalty applied to the specified index.
        apply_penalty (bool): Whether to apply the penalty to the specified index. Default is True.

    Returns:
        torch.Tensor: A tensor containing class weights, where each weight corresponds to a label.

    Notes:
        - The weights are computed based on the inverse of label percentages in the training set.
        - If apply_penalty is True, a penalty is applied to the weight of the label at the specified index.
    """
    labels = training_set['labels']
    n = labels.size()[0]
    labels_list = labels[labels != -1].flatten().tolist()
    percentages = { l:Counter(labels_list)[l] / float(n) for l in range(num_labels) }
    weights = [ 1-percentages[l] if l in percentages else 0.0 for l in list(range(num_labels)) ]
    weights = torch.tensor(weights, device=device)

    # add further penalty to the index class
    if apply_penalty:
        weights[index] = weights[index]/penalty

    return weights


def load_emotions_classifier(sm):
    """
    Load a pretrained emotions classifier model.

    Args:
        sm (str): The specific model identifier or name to load.

    Returns:
        emotionsClassifier: The pretrained emotions classifier model loaded from file.
    """
    emotionsClassifier = torch.load(f'../models/best_model_{sm}.pt')
    return emotionsClassifier


def resample_dataframe(df):
    """
    The resampling function used in the utterance datasets used to pre-train the emotion classifier for Sentence BERT model. See format_utt_sentence_dataset() in preprocessing.py

    Args
        df (DataFrance): the dataframe to resample

    Returns
        final_df (DataFrame): the resampled dataframe
    """

    df_0 = df[df['lab_val']==0]
    df_1 = df[df['lab_val']==1]
    df_2 = df[df['lab_val']==2]
    df_3 = df[df['lab_val']==3]
    df_4 = df[df['lab_val']==4]
    df_5 = df[df['lab_val']==5]
    df_6 = df[df['lab_val']==6]

    maj_class0 = resample(df_0, 
                        replace=True,     
                        n_samples=2000,    
                        random_state=123) 
    maj_class1 = resample(df_1, 
                        replace=True,     
                        n_samples=500,    
                        random_state=123) 
    maj_class2 = resample(df_2, 
                        replace=True,     
                        n_samples=500,    
                        random_state=123) 
    maj_class3 = resample(df_3, 
                        replace=True,     
                        n_samples=300,    
                        random_state=123) 
    maj_class4 = resample(df_4, 
                        replace=True,     
                        n_samples=2000,    
                        random_state=123) 
    maj_class5 = resample(df_5, 
                        replace=True,     
                        n_samples=300,    
                        random_state=123) 
    maj_class6 = resample(df_6, 
                        replace=True,     
                        n_samples=1000,    
                        random_state=123) 

    final_df = pd.concat([maj_class0, maj_class1, maj_class2, maj_class3, maj_class4, maj_class5, maj_class6])
    return final_df

# AUXILIARY FUNCTIONS FOR PREPROCESSING

def get_and_set_pretrained_vectors():
    """
    Load the pretrained vectors from FastText and process some steps to consider padding and handle OOV tokens.

    Returns:
        pretrained_vectors: the FastText pretrained vectors
        mapping_dict
    """
    pretrained_vectors = FastText(language='en')
    pretrained_vocab = vocab(pretrained_vectors.stoi)
    unk_index, pad_index = 0, 1
    pretrained_vocab.insert_token("<unk>", unk_index)
    pretrained_vocab.insert_token("<pad>", pad_index)
    pretrained_vocab.set_default_index(unk_index)
    vocab_itos = pretrained_vocab.get_itos()
    vocab_stoi = pretrained_vocab.get_stoi()

    return pretrained_vectors, vocab_itos, vocab_stoi


def map_tokens_to_vectors(entry, pretrained_vectors, vocab_itos):
    """
    /!\ THIS IS NOT OPTIMIZED SINCE IT COMPUTES TO VOC FOR EACH ENTRY! TO BE SOLVED

    Map the output values of dailydialog text after applying tweet tokenizer to their associated vectors from FastText embeddings

    Args:
        entry: an instance of dailydialog dataset, containing both text and labels

    Returns:
        dict: a dictionary with the processed instances and the original utterances and labels
    """
    label = entry['label']
    indexes = entry['text']
    vectors = []
    for index in indexes:
        try:
            # retrieve the token with such index
            token = vocab_itos[index]
            # retrive the vector for such token
            vector = pretrained_vectors.get_vecs_by_tokens(token)
        except:
            # generate a random tensor of the same size
            vector = torch.normal(mean=0, std=0.3, size=(1,300)).view(300)

        vectors.append(vector)

    # deal with the particular case of test set to store indexes
    return {'label': label, 'text': indexes, 'vectors': vectors}


def add_sentencebert_random_vectors(embedding, size, max_length):
    """
    This auxiliary function is to be used for Sentence BERT preprocessing. It adds the number of 384-sized random vectors defined by the parameter size.

    Args:
        size (int): the number of random vectors to generate.

    Returns:
        vectors (list): the list of length size of generated random vectors.
    """
    # DescribeResult(nobs=3840, minmax=(-0.20482617616653442, 0.17243704199790955), mean=0.00021893562853630052, variance=0.0026047970850628303, skewness=-0.06372909078235393, kurtosis=-0.001260392396507104)
    mean = 0.00021893562853630052
    variance = 0.0026047970850628303
    for _ in range(size):
        embedding.append([random.gauss(mu=mean, sigma=sqrt(variance)) for _ in range(max_length)])
    return embedding


# AUXILIARY FUNCTIONS FOR QUALITATIVE ANALYSIS

def get_dyda_test_entries(examples=20):
    # Emotion distrib. in test set : {0: 6321, 4: 1019, 1: 118, 6: 116, 5: 102, 2: 47, 3: 17})
    """
    This is an auxiliary function for qualitative analysis. It randomly samples entries from dailydialog test set. At this stage, dailydialog test set is supposed to be indexed through an 'index' column. The random sampling is performed such as all the emotions should be sampled at least twice.

    Args:
        examples (int): the number of examples we want to sample (default=20)

    Returns:
        indexes (list): a list of the indexes of the sampled entries, whose length is equal to examples parameter
    """
    # load initial dataset
    dailydialog = load_dataset('daily_dialog')
    # index test set and store result in a new variable
    dailydialog['test'] = test_set_with_indexes(dailydialog['test'])
    test_data = dailydialog['test']

    sampled_examples = 0
    indexes = []
    emotions_amounts = [0 for _ in range(7)] # count how many times each emotion is represented
    represented_emotions = [False for _ in range(7)] # set to True when 2 samples of this emotion are sampled
    n_examples = len(test_data['dialog'])

    # since an emotion is not yet enough represented
    while False in represented_emotions:
        # pick a dialog and retrieve its unique emotions
        picked_example = random.randint(1, n_examples)
        emotions_in_picked_example = list(set(test_data[picked_example]['emotion']))

        # while the examples covers only enough represented emotion, we sample again
        while False not in [represented_emotions[i] for i in emotions_in_picked_example]:
            picked_example = random.randint(1, n_examples)
            emotions_in_picked_example = list(set(test_data[picked_example]['emotion']))

        # update emotions amounts
        for emotion in test_data[picked_example]['emotion']:
            emotions_amounts[emotion] += 1

        # update represented emotions
        for emotion in range(len(emotions_amounts)):
            if emotions_amounts[emotion] >= 2 and represented_emotions[emotion] == False:
                represented_emotions[emotion] = True

        indexes.append(picked_example)

    return indexes


def test_set_with_indexes(test_data, separate_utterances=False, max_dialog_length=12):
    """
    This is an auxiliary function for qualitative analysis. This function aims at indexing the whole test set used at test step in order to retrieve the original examples when they are processed by the model. Due to random shuffling when instanciating DataLoader, the index is necessary to associate a dialog to a prediction.

    Args:
        test_data (DatasetDict): the original DailyDialog test set
        separate_utterances (bool): default=False. This indicates if we further need to process data in an isolated utterance setting (separate_utterances=True) or a contextual utterance setting (separate_utterances=False)
        max_dialog_length (int): default=12. Dialog padding length, intended to be the same that the one given in the preprocessing pipeline

    Returns:
        new_test_data (DatasetDict):   the indexed test dataset
    """
    # contextual representations -> one label per dialog
    if not separate_utterances:
        new_test_data = Dataset.from_dict({'dialog':test_data['dialog'], 'act': test_data['act'],'emotion':test_data['emotion'], 'index':range(1,1001)})
    # indiv utt representations -> one label per utterance
    else:
        ref = test_data['dialog']
        index_vectors = [[i+1 for _ in range(max_dialog_length)] for i in range(len(ref))]
        new_test_data = Dataset.from_dict({'dialog':test_data['dialog'], 'act': test_data['act'],'emotion':test_data['emotion'], 'index':index_vectors})
    return new_test_data


def find_triplet_indexes(emotions):
    """
    This in an auxialiary function for static utterance qualitative analysis. This function takes a tripletable list and returns a possible triplet with its indexes and associated emotions in the precise order (A, P, N)

    Args:
        emotions (list): a list of emotion indexed labels that allows to make triplets from it

    Returns:
        A (int): anchor sample
        P (int): positive sample
        N (int): negative sample
        index_A (int): anchor index in emotions
        index_P (int): positive index in emotions
        index_N (int): negative index in emotions
    """
    A, P, N = 0, 0, 0
    index_A, index_P, index_N = 0, 0, 0
    countdown = Counter(emotions)
    count = sorted(countdown, key=countdown.get, reverse=True)
    A = count[0]
    P = A
    N = count[1]
    
    for i in range(len(emotions)):
        if emotions[i] == A:
            index_A = i
        if emotions[i] == N:
            index_N = i
    for i in range(len(emotions)):
        if emotions[i] == A and i != index_A:
            index_P = i
    return A, P, N, index_A, index_P, index_N


def find_tripletable_indexes(dailydialog):
    """
    This in an auxialiary function for static utterance qualitative analysis. This function considers all DailyDialog test set data samples and returns all the indexes of tripletable data samples. A sample is said to be tripletable if its emotions labels can be used to make a triplet regarding triplet loss setting.

    Args:
        dailydialog (DatasetDict): the original DailyDialog dataset

    Returns:
        tripletable (list): a list of data sample indexes
    """
    tripletable = []
    for i in range(len(dailydialog['test'])):
        emotions = dailydialog['test'][i]['emotion']
        count = Counter(emotions)
        if count.most_common(1)[0][1] >= 2 and len(list(count)) >= 2:
            tripletable.append(i)
    return tripletable


def activate_gpu(force_cpu=False):
    """
    Manage the device to use for computations, either cpu, mps (for Mac M1 and further chips) or cuda.

    Args:
        force_cpu (bool): force the selected device being cpu

    Returns:
        device (str): the name of the selected device
    """
    device = "cpu"
    if not force_cpu:
        if torch.cuda.is_available():
            device = 'cuda'
            print('DEVICE = ', colored(torch.cuda.get_device_name(0), "green" ) )
        elif torch.backends.mps.is_available():
            device = 'mps'
            print('DEVICE = ', colored("mps", "green" ) )
        else:
            device = 'cpu'
            print('DEVICE = ', colored('CPU', "blue"))
    return device


def sep_indexes_from_batch(x, sep_id):
    """
    -------------------------------- /!\ DEPRECATED (but still used in models.py) /!\ --------------------------------
    This function is using list format to process data. This is no longer useful because converting tensor to list implies to loose the gradient, which is not suitable in our case.
    ------------------------------------------------------------------------------------------------------------------

    Args:
        x (tensor): input tensor of the foward loop, thus containing the input ids
        sep_id (int): the value of the index for the SEP token

    Returns:
        sep_indexes (list): a list of lists containing the list of SEP indexes for each dialog of the batch
    """
    # convert tensor to list for easier data manipulation
    input_ids = x.tolist()
    sep_indexes = [[] for _ in range(len(input_ids))]

    # iterate through dialogs
    for i in range(len(input_ids)):
        dialog = input_ids[i]

        # iterate through utterances representations
        for j in range(len(dialog)):
            if dialog[j] == sep_id:
                sep_indexes[i].append(j)

    return sep_indexes

def get_quali_static_arrays(dailydialog, sae, cdh, csh):
    """
    This is an auxialiary function for qualitative experiments. Here is a reminder of the acronyms meaning:
        - SAE = Still An Emotion. Wrong predictions where ground truth label was 'no emotion', but the utterance still conveys a feeling that is not described by DailyDialog labels
        - CDH = Context Did Help. Correct predictions where the dialog context seems to have help produce the right result.
        - CSH = Context Should Help. Incorrect prediction where the dialog context should have helped in producing the right emotion.

    This function takes the indexes of the relevant examples depending on what they represent and then output the triplet arrays so that the trained model can further be used to infer on these examples.

    Args:
        dailydialog:             the original dailydialog dataset
        sae (list):              some relevant sae example indexes
        cdh (list):              some relevant sae example indexes
        csh (list):              some relevant sae example indexes

    Returns:
        sae_arr (list):         a list of triplets (tuples) to enable inference on sae examples
        cdh_arr (list):         a list of triplets (tuples) to enable inference on cdh examples
        csh_arr (list):         a list of triplets (tuples) to enable inference on csh examples
    """

    # store indexes of relevant examples
    sae_utt_id = [2, 3, 5, 7, 11]
    cdh_utt_id = [3, 5, 8, 11]
    csh_utt_id = [9, 3, 2, 4, 2, 11]

    # the values of indexes in this array are hard coded for reproducibility concerns. Indeed, it is important to proceed the experiments relying on the same utterances than in the conversational case, in order to fully compare the results

    sae_arr = [(dailydialog['test'][sae[0]]['emotion'][sae_utt_id[0]], dailydialog['test'][sae[0]]['emotion'][4], dailydialog['test'][sae[0]]['emotion'][0], sae_utt_id[0], 4, 0), 
            (dailydialog['test'][sae[1]]['emotion'][sae_utt_id[1]], dailydialog['test'][sae[1]]['emotion'][0], dailydialog['test'][sae[1]]['emotion'][5], sae_utt_id[1], 0, 5),
            (dailydialog['test'][sae[2]]['emotion'][sae_utt_id[2]], dailydialog['test'][sae[2]]['emotion'][6], dailydialog['test'][sae[2]]['emotion'][3], sae_utt_id[2], 6, 3),
            (dailydialog['test'][sae[3]]['emotion'][sae_utt_id[3]], dailydialog['test'][sae[3]]['emotion'][8], dailydialog['test'][sae[3]]['emotion'][0], sae_utt_id[3], 8, 0),
            (dailydialog['test'][sae[4]]['emotion'][sae_utt_id[4]], dailydialog['test'][sae[4]]['emotion'][13], dailydialog['test'][sae[4]]['emotion'][2], sae_utt_id[4], 13, 2)]

    cdh_arr = [(dailydialog['test'][cdh[0]]['emotion'][4], dailydialog['test'][cdh[0]]['emotion'][cdh_utt_id[0]], dailydialog['test'][cdh[0]]['emotion'][5], 4, cdh_utt_id[0], 5), 
            (dailydialog['test'][cdh[1]]['emotion'][7], dailydialog['test'][cdh[1]]['emotion'][cdh_utt_id[1]], dailydialog['test'][cdh[1]]['emotion'][6], 7, cdh_utt_id[1], 6),
            (dailydialog['test'][cdh[2]]['emotion'][9], dailydialog['test'][cdh[2]]['emotion'][cdh_utt_id[2]], dailydialog['test'][cdh[2]]['emotion'][10], 9, cdh_utt_id[2], 10),
            (dailydialog['test'][cdh[3]]['emotion'][10], dailydialog['test'][cdh[3]]['emotion'][cdh_utt_id[3]],  dailydialog['test'][cdh[3]]['emotion'][3], 10, cdh_utt_id[3], 3)]

    csh_arr = [(dailydialog['test'][csh[0]]['emotion'][10], dailydialog['test'][csh[0]]['emotion'][csh_utt_id[0]], dailydialog['test'][csh[0]]['emotion'][11], 10, csh_utt_id[0], 11), 
            (dailydialog['test'][csh[1]]['emotion'][4], dailydialog['test'][csh[1]]['emotion'][csh_utt_id[1]], dailydialog['test'][csh[1]]['emotion'][5], 4, csh_utt_id[1], 5),
            (dailydialog['test'][csh[2]]['emotion'][3], dailydialog['test'][csh[2]]['emotion'][csh_utt_id[2]], dailydialog['test'][csh[2]]['emotion'][4], 3, csh_utt_id[2], 4),
            (dailydialog['test'][csh[3]]['emotion'][5], dailydialog['test'][csh[3]]['emotion'][csh_utt_id[3]], dailydialog['test'][csh[3]]['emotion'][6], 5, csh_utt_id[3], 6),
            (dailydialog['test'][csh[4]]['emotion'][3], dailydialog['test'][csh[4]]['emotion'][csh_utt_id[4]], dailydialog['test'][csh[4]]['emotion'][4], 3,csh_utt_id[4], 4),
            (dailydialog['test'][csh[5]]['emotion'][10], dailydialog['test'][csh[5]]['emotion'][csh_utt_id[5]], dailydialog['test'][csh[5]]['emotion'][0], 10, csh_utt_id[5], 0)]
    
    return sae_arr, cdh_arr, csh_arr