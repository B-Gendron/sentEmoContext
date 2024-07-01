import numpy as np
from math import sqrt
from datasets import load_dataset
import torch
from torchmetrics.classification import MulticlassMatthewsCorrCoef
import string
from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score

dailydialog = load_dataset('daily_dialog')


def format_prompt_from_dialog(split, dial_id, utt_id):
    """
    This auxiliary function concats the dialog utterances in a string by adding hyphens and line breaks at each utterance separation.

    Args:
        dialog (list): a dialog represented as a list of utterances

    Returns:
        one_line (str): a string containing the dialog content and some formatting characters
    """
    dialog = dailydialog[split][dial_id]['dialog']
    utterance = dialog[utt_id]
    header = "Here is a dialog: \n"
    one_line = '\n- ' + '\n-'.join(dialog)
    footer = f"\n\nRegarding its conversational context, give the appropriate emotion to describe this utterance: '{utterance}', amongst: happiness, sadness, anger, surprise, fear, disgust. If none of them seems to correspond, the appropriate answer is no emotion. In this case, the most appropriate emotion label is:"
    return header + one_line + footer


def format_prompt_last_utterance(split, dial_id):
    """
    This auxiliary function concats the dialog utterances in a string by adding hyphens and line breaks at each utterance separation.

    Args:
        dialog (list): a dialog represented as a list of utterances

    Returns:
        one_line (str): a string containing the dialog content and some formatting characters
    """
    dialog = dailydialog[split][dial_id]['dialog']
    header = "Here is a dialog: \n"
    one_line = '\n- ' + '\n-'.join(dialog)
    footer = f"\n\nRegarding its conversational context, give the appropriate emotion for the last utterance among: happiness, sadness, anger, surprise, fear, disgust. If none of them seems to correspond, the appropriate answer is no emotion. In this case, the most appropriate emotion label is:"
    return header + one_line + footer


def format_prompt_last_utterance_falcon(split, dial_id):
    """
    This auxiliary function concats the dialog utterances in a string by adding hyphens and line breaks at each utterance separation.

    Args:
        dialog (list): a dialog represented as a list of utterances

    Returns:
        one_line (str): a string containing the dialog content and some formatting characters
    """
    dialog = dailydialog[split][dial_id]['dialog']
    header = "Here is a dialog: \n"
    one_line = '\n- ' + '\n-'.join(dialog)
    footer = f"\n\nRegarding its conversational context, return the appropriate emotion for the last utterance among: sadness, happiness, anger, surprise, fear and disgust. If none of them properly correspond, return 'no emotion'."
    return header + one_line + footer


def minimum_index(l):
    """
    A simple auxiliary function that returns the index of the minimum value in a list.

    Args:
        l (list): the list to consider
    
    Returns:
        min_index (int): the index of the minimum value in list l
    """
    min_index = 0
    list_len = len(l)
    for index in range(list_len):
        if l[index] < l[min_index]:
            min_index = index
    return min_index


def predicted_emotion(output):
    """
    Use the word_in_string auxiliary function to find the predicted emotion in Llama output.

    Args:
        output (str): the text produced by the LLM

    Returns:
        emotion (str): the emotion given by the LLM output
    """
    emotions = ['happiness', 'fear', 'anger', 'disgust', 'surprise', 'sadness', 'no emotion']
    indexes_list = [len(output)+1 for _ in range(len(emotions))]
    truncated_output = output[output.find("is:"):]
    splitted_output = truncated_output.translate(str.maketrans('', '', string.punctuation)).split()
    new_output = ' '.join(splitted_output)
    for i in range(len(emotions)):
        word = emotions[i]
        find_index = new_output.find(word)
        if find_index > -1:
            indexes_list[i] = find_index
    return emotions[minimum_index(indexes_list)]


def map_emotion_to_index(emotion):
    """
    An auxiliary function that converts an emotion label to its associated index according to dailydialog construction.

    Args:
        emotion (str): the emotion label. Must be one of these: no emotion, happiness, anger, sadness, surprise, fear, disgust.

    Returns:
        index (int): an integer between 0 and 6 that represent the emotion label in dailydialog.
    """
    labels = {'no emotion':"0", 'anger':"1", 'disgust':"2", 'fear':"3", 'happiness':"4", 'sadness':"5", 'surprise':"6"}
    return int(labels[emotion])


def custom_flatten(ll):
    """
    A function to flatten a list of lists where sub lists are of heterogeneous sizes.

    Args:
        ll (list): the input list of lists

    Returns:
        l (list): the flattened list   
    """
    l = []
    for sl in ll:
        l.extend(sl)
    return l


def compute_metrics_and_variance(all_trues, all_preds):
    """
    This function computes and stores the following metrics from the LLM inferences:
        - MCC (using Torch and Sklearn utils)
        - Micro, Macro and Weighted F1
    It outputs the mean and variance of the metrics over the episodes.

    Args:
        all_trues (list):       the gold labels
        all_preds (list):       the predicted labels
    
    Returns:
        results (dict):        the mean and variance of abovementioned metrics
    """
    mcc_torch_list, mcc_list, macro_list, micro_list, f1_list = [], [], [], [], []

    # dump preds and trues

    for i in range(len(all_trues)):
        trues, preds = all_trues[i], all_preds[i]

        # compute classification metrics on each run
        metric = MulticlassMatthewsCorrCoef(num_classes=7)
        mcc_torch = metric(torch.Tensor(preds), torch.Tensor(trues))
        mcc = matthews_corrcoef(trues, preds)
        microf1 = f1_score(trues, preds, average='micro', labels=[1,2,3,4,5,6])
        macrof1 = f1_score(trues, preds, average='macro', labels=[1,2,3,4,5,6])
        f1 = f1_score(trues, preds, average='weighted', labels=[1,2,3,4,5,6])
        cm = confusion_matrix(trues, preds)

        # store values in corresponding lists
        mcc_torch_list.append(mcc_torch.item())
        mcc_list.append(mcc)
        micro_list.append(microf1)
        macro_list.append(macrof1)
        f1_list.append(f1)
    
    std_mcc_torch, mean_mcc_torch = np.std(mcc_torch_list), np.mean(mcc_torch_list)
    std_mcc, mean_mcc = np.std(mcc_list), np.mean(mcc_list)
    std_micro, mean_micro = np.std(micro_list), np.mean(micro_list)
    std_macro, mean_macro = np.std(macro_list), np.mean(macro_list)
    std_f1, mean_f1  = np.std(f1_list), np.mean(f1_list)

    results = {'mean' : [mean_mcc_torch, mean_mcc, mean_micro, mean_macro, mean_f1], 'std' : [std_mcc_torch, std_mcc, std_micro, std_macro, std_f1], 'cm': cm}

    return results


def store_classification_metrics(results, model):
    """
    This function writes the output of `compute_metrics_and_variance()` into a .txt file stored in a ./results/ folder.

    Args:
        results (dict):          the output of `compute_metrics_and_variance()`
        model (str):             the model name
    """
    means = results['mean']
    std = results['std']

    with open(f"./results/falcon-7b-last.txt", "a") as f:
        print(f"CLASSIFICATION SCORES FOR {model} ON DAILYDIALOG TEST SET\n",file=f)
        print("Confusion matrix\n", results['cm'], file=f)
        print('MCC (Torch): ', means[0], '+/-', std[0], file=f)
        print('MCC (Sklearn): ', means[1], '+/-', std[1], file=f)
        print('Micro F1: ', means[2], '+/-', std[2], file=f)
        print('Macro F1: ', means[3], '+/-', std[3], file=f)
        print('Weighted F1: ', means[4], '+/-', std[4], file=f)


def compute_mcc_from_cm(TP, TN, FP, FN):
    """
    A simple computation of MMC using the elements of the confusion matrix in a binary classification scenario.

    Args:
        TP (int): the number of true positives
        TN (int): the number of true negatives
        FP (int): the number of false positives
        FN (int): the number of false negatives

    Returns:
        float: the value of the MCC
    """
    top = TP*TN - FP*FN
    bottom = (TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)
    return top / sqrt(bottom)