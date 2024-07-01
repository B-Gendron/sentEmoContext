# Torch utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassMatthewsCorrCoef
torch.set_default_dtype(torch.float32)

# Data loading and preprocessing
from datasets import load_dataset, Dataset
from transformers import logging
logging.set_verbosity_error()

# General purposes modules
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score
import random
random.seed(42)
from tqdm import tqdm
from termcolor import colored
import logging
import warnings
warnings.filterwarnings("ignore")

# From others files of this repo
from utils import *
from models import EmotionalLabelsClassifier
from datasets_classes import SentenceEmotionDatasetUtterances

# 🛑 Tensorboard
from torch.utils.tensorboard import SummaryWriter


def get_args_and_dataloaders(dataset):
    """
    Instantiate the training hyperparameters and the dataloaders.

    Args:
        dataset (DatasetDict): a huggingface DatasetDict object, supposed already preprocessed

    Returns:
        args (dict): a dictionary that containes the hyperparameters for training
        train_loader (dataloader): the dataloader that contains the training samples
        val_loader (dataloader): the dataloader that contains the validation samples
        test_loader (dataloader): the dataloader that contains the test samples
    """
    args = {'train_bsize': 16, 'eval_bsize': 8, 'lr': 0.0005, 'max_eps':50}

    # num_samples = None
    # train_batch_sampler = ImbalancedDatasetSampler(SentenceEmotionDatasetUtterances(dataset["train"], args), level='utterance', num_samples=num_samples, spreading=False)
    
    train_loader = DataLoader(dataset=SentenceEmotionDatasetUtterances(dataset["train"], args=args), shuffle=True, batch_size=args['train_bsize'], drop_last=True)
    val_loader   = DataLoader(dataset=SentenceEmotionDatasetUtterances(dataset["validation"], args=args), batch_size=args['eval_bsize'],shuffle=True, drop_last=True)
    test_loader  = DataLoader(dataset=SentenceEmotionDatasetUtterances(dataset["test"], args=args), batch_size=args['eval_bsize'], shuffle=True, drop_last=True)
    return args, train_loader, val_loader, test_loader


def train(args, model, device, train_loader, optimizer, epoch, writer):
    """
    Perfom one epoch of model training in the case of a regular model trained directly on the triplet loss.

    Args:
        args (str): the hyperparameters for the training
        model: the model to train
        device (str): the device where to send the inputs and the model
        train_loader (dataloader): the dataloader that contains the training samples
        optimizer: the optimizer to use for training
        epoch (int): the index of the current epoch
        writer: for tensorboard logging

    Returns:
        loss_it (list): the list of all the losses on each batch for the epoch
    """
    model.to(device)
    model.train()
    loss_it = []
    trues, preds = [], []

    for it, batch in tqdm(enumerate(train_loader), desc="Epoch %s: " % (epoch+1), total=train_loader.__len__()):
        batch = {'embedding': batch['embedding'].to(device, dtype = torch.float32), 'labels': batch['label'].to(device, dtype = torch.float32)}
        optimizer.zero_grad()

        # perform training
        classes_probas = model(batch['embedding'])
        labels = batch['labels'].squeeze(-1).long()
        weights = get_labels_weights(batch, device, num_labels=7, apply_penalty=False).to(device)
        ce_loss = nn.CrossEntropyLoss(weight=weights)
        loss = ce_loss(classes_probas, labels)
        loss.backward()
        optimizer.step()

        # store loss history
        loss_it.append(loss.item())
        _, pred  = torch.max(classes_probas, 1)
        preds.append(pred)
        trues.append(labels)

    loss_it_avg = sum(loss_it)/len(loss_it)

	# 🛑 add some metrics to keep with a label and the epoch index
    writer.add_scalar("Loss/train", loss_it_avg, epoch)

    trues, preds = torch.cat(trues).flatten().tolist(), torch.cat(preds).flatten().tolist()
    metric = MulticlassMatthewsCorrCoef(num_classes=7)
    mcc = metric(torch.Tensor(preds), torch.Tensor(trues)).item()
    precision = precision_score(trues, preds, average='weighted', zero_division=0)
    recall = recall_score(trues, preds, average="weighted")
    f1score = f1_score(trues, preds, average='weighted')

    print("Epoch %s/%s - %s : (%s %s) (%s %s) (%s %s) (%s %s) (%s %s)" % (colored(str(epoch+1), 'blue'), args['max_eps'] , colored('Tr. encoder', 'blue'), colored('Average CE loss: ', 'cyan'), loss_it_avg, colored('MCC: ', 'cyan'), mcc, colored('Precision: ', 'cyan'), precision, colored('Recall: ', 'cyan'), recall, colored('F1: ', 'cyan'), f1score))

    return loss_it_avg


def test(target, model, loader, device):
    """
    Perfom model evaluation on the validation or test set for one epoch.

    Args:
        target (str):         indicates on which set the model is evaluated, for a coherent display
        model:                the model to evaluate
        loader (DataLoader):  the dataloader, either validation or test loader
        device (str):         the device where to send the inputs and the model

    Returns:
        loss_it_avg (float): the average value of the loss on all the batches for one epoch 
    """
    model.eval()
    loss_it = []
    preds, trues = [], []

    for it, batch in tqdm(enumerate(loader), desc="%s: " % (target), total=loader.__len__()):

        with torch.no_grad():
            
            batch = {'embedding': batch['embedding'].to(device, dtype = torch.float32), 'labels': batch['label'].to(device, dtype = torch.float32)}

            # perform training
            classes_probas = model(batch['embedding'])
            labels = batch['labels'].squeeze(-1).long()
            ce_loss = nn.CrossEntropyLoss()
            loss = ce_loss(classes_probas, labels) 
            loss_it.append(loss.item())
 
            _, pred  = torch.max(classes_probas, 1)
            trues.append(labels)
            preds.append(pred)

    trues, preds = torch.cat(trues).flatten().tolist(), torch.cat(preds).flatten().tolist()
    metric = MulticlassMatthewsCorrCoef(num_classes=7)
    mcc = metric(torch.Tensor(preds), torch.Tensor(trues)).item()
    precision = precision_score(trues, preds, average='weighted', zero_division=0)
    recall = recall_score(trues, preds, average="weighted")
    macro_f1 = f1_score(trues, preds, average='macro', labels=[1,2,3,4,5,6])
    micro_f1 = f1_score(trues, preds, average='micro', labels=[1,2,3,4,5,6])
    f1score = f1_score(trues, preds, average='weighted', labels=[1,2,3,4,5,6])
    val_loss_avg = sum(loss_it)/len(loss_it)


    # affichage scores après CE : loss, MCC, Precision, Recall, F1
    print("%s: (%s %s) (%s %s) (%s %s) (%s %s) (%s %s) (%s %s) (%s %s)" % (colored(target, 'blue'), colored('Average CE loss: ', 'cyan'), val_loss_avg, colored('MCC: ', 'cyan'), mcc, colored('Precision: ', 'cyan'), precision, colored('Recall: ', 'cyan'), recall, colored('Macro F1: ', 'cyan'), macro_f1, colored('Micro F1: ', 'cyan'), micro_f1, colored('F1: ', 'cyan'), f1score))

    return val_loss_avg, mcc, precision, recall, macro_f1, micro_f1, f1score


def run_epochs(model, args, optimizer, train_loader, val_loader, test_loader, device, writer, sm):
    """
    Perform the training and evaluate the model on the validation set for each epoch.

    Args:
        model: the model to train and evaluate
        args (dict): the hyperparameters for the training
        optimizer: the optimizer to use for training
        train_loader (DataLoader): the dataloader that contains training samples
        train_loader (DataLoader): the dataloader that contains validation samples
        train_loader (DataLoader): the dataloader that contains test samples
        device (str): the device where to send the inputs and the model
        writer: for tensorboard logging
        sm: the sentence transformer model to use

    Returns:
        val_losses (list): a list that contains the validation loss values for each epoch
    """
    val_losses = []
    biggest_mcc = 0
    best_model = model

    for ep in range(args['max_eps']):
        # train the model on the train loader
        train_loss = train(args, model, device, train_loader, optimizer, ep, writer)
        # infer on the validation loader
        val_loss_avg, mcc, precision, recall, macro_f1, micro_f1, f1score = test("validation", model, val_loader, device)
        # 🛑
        writer.add_scalar("Loss/val", val_loss_avg, ep)
        writer.add_scalar("MCC/val", mcc, ep)
        writer.add_scalar("F1/val", f1score, ep)
		# append the validation losses (good losses should normally go down)
        val_losses.append(val_loss_avg)

        # log the validation metrics
        val_metrics = {"Loss/train": round(train_loss, 4), "Loss/val": round(val_loss_avg, 4), "MCC/val":round(mcc, 4)}

        # check if this epoch's model is the best so far
        if mcc > biggest_mcc:
            biggest_mcc = mcc
            best_model = model

    print(colored(f'Best model metrics: MCC {biggest_mcc}, Prec. {precision}, Rec. {recall}, Macro F1 {macro_f1}, Micro F1 {micro_f1}, F1 score {f1score}', 'green'))
    # save the best model
    torch.save(best_model.state_dict(), f'../models/best_model_{sm}.pt')

    test('test', model, test_loader, device)

	# 🛑 flush to perform all remaining operations
    writer.flush()

    # return the list of epoch validation losses in order to use it later to create a plot
    return val_losses


def run_exp(sm, device):
    """
    Run the pre-training of the emotion classifier.

    Args:
        sm: the sentence transformer model used to generate utterance embeddings
        device: the device on which computations should be perfomed
    """
    emotional_utterances = load_dataset(f'all_balanced_utterances_{sm}')
    args, train_loader, val_loader, test_loader = get_args_and_dataloaders(emotional_utterances)

    model = EmotionalLabelsClassifier(args, device, max_lengths[f'{sm}'])
    optimizer_ce = optim.Adam(model.parameters(), lr=args['lr'])
    writer = SummaryWriter(f"{experiment}/{get_datetime()}", comment=args2filename(args))
    run_epochs(model, args, optimizer_ce, train_loader, val_loader, test_loader, device, writer, sm)


if __name__ == "__main__":

    # Instantiate parser and retrieve arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", type=str, help="give a name to the experiment folder that stores Tensorflow logs", default="unnamed_experiment")
    parser.add_argument("-p", "--pretrained", type=str, help="the pretrained model to use between 'mpnet', 'roberta' and 'minilm'.", default='all')
    arguments = parser.parse_args() 
    experiment = arguments.exp
    pretrained_mapping = {'minilm':'all-MiniLM-L6-v2', 'mpnet':'all-mpnet-base-v2', 'roberta':'all-roberta-large-v1'}

    # Setup experiment settings
    max_lengths = {'all-MiniLM-L6-v2':384, 'all-roberta-large-v1':1024, 'all-mpnet-base-v2':768}
    sentence_models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'all-roberta-large-v1']
    device = activate_gpu() 
    model_count = 1

    if arguments.pretrained in ['mpnet', 'minilm', 'roberta']:
        pretrained = pretrained_mapping[arguments.pretrained]
        print(colored(f"Model: {pretrained}", 'yellow'))
        run_exp(pretrained, device)

    elif arguments.pretrained == 'all':
        for pretrained in pretrained_mapping.values():
            print(colored(f"Model: {pretrained}", 'yellow'))
            run_exp(pretrained, device)

    else:
        print(colored("Invalid argument for pretrained model! Valid arguments are 'minilm', 'mpnet', 'roberta', and 'all' to choose all of them.", 'red'))