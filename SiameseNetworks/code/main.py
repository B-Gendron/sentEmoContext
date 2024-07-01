# Torch utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float32)

# Data loading and preprocessing
from datasets import load_dataset
import datasets
from transformers import logging
logging.set_verbosity_error()

# General purposes modules
import argparse
import numpy as np
import csv
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
import random
random.seed(42)
from tqdm import tqdm
from termcolor import colored
import logging
import warnings
warnings.filterwarnings("ignore")

# From others files of this repo
from models import SiameseNetworkLSTM, SentenceEmCoBERT
from utils import *
from datasets_classes import UtteranceVectorsEmotionDataset, SentenceEmotionDatasetBERT

# 🛑 Tensorboard
from torch.utils.tensorboard import SummaryWriter


def get_args_and_dataloaders(dataset, dataset_class):
    """
    Instantiate the training hyperparameters and the dataloaders.

    Args:
        dataset (Dataset): The data to put in the DataLoader.
        dataset_class (Dataset): The consistent dataset class from the datasets.py script to process data.

    Returns:
        args (dict): A dictionary that contains the hyperparameters for training.
        train_loader (DataLoader): The dataloader that contains the training samples.
        val_loader (DataLoader): The dataloader that contains the validation samples.
        test_loader (DataLoader): The dataloader that contains the test samples.
    """
    args = {'train_bsize': 32, 'eval_bsize': 1, 'lr': 0.00001, 'spreading':False}
    train_loader = DataLoader(dataset=dataset_class(dataset["train"], args=args), pin_memory=True, batch_size=args['train_bsize'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(dataset=dataset_class(dataset["validation"], args=args), pin_memory=True, batch_size=args['eval_bsize'], shuffle=True, drop_last=True)
    test_loader  = DataLoader(dataset=dataset_class(dataset["test"], args=args), pin_memory=True, batch_size=args['eval_bsize'], shuffle=True, drop_last=True)
    return args, train_loader, val_loader, test_loader


def get_args_and_dataloaders_balanced(dataset, dataset_class, spreading):
    """
    Instantiate the training hyperparameters and the dataloaders using some sampling to balance data across emotion classes.

    Args:
        dataset (Dataset): The data to put in the DataLoader.
        dataset_class (Dataset): The consistent dataset class from the datasets.py script to process data.

    Returns:
        args (dict): A dictionary that contains the hyperparameters for training.
        train_loader (DataLoader): The dataloader that contains the training samples.
        val_loader (DataLoader): The dataloader that contains the validation samples.
        test_loader (DataLoader): The dataloader that contains the test samples.
    """
    args = {'train_bsize': 32, 'eval_bsize': 1, 'lr': 0.00001, 'spreading':spreading}

    num_samples = None
    train_batch_sampler = ImbalancedDatasetSampler(dataset_class(dataset["train"], args), num_samples=num_samples, spreading=spreading)

    train_loader = DataLoader(dataset_class(dataset["train"], args), pin_memory=True, sampler=train_batch_sampler, batch_size=args['train_bsize'])
    val_loader = DataLoader(dataset_class(dataset["validation"], args), pin_memory=True, batch_size=args['eval_bsize'])
    test_loader = DataLoader(dataset_class(dataset["test"], args), pin_memory=True, batch_size=args['eval_bsize'])

    return args, train_loader, val_loader, test_loader


def train_isolated(args, model, train_loader, optimizer, epoch):
    """
    Perform one epoch of model training for the isolated utterance model trained directly on the triplet loss.

    Args:
        args (str): The hyperparameters for the training.
        model: The model to train.
        device (str): The device where to send the inputs and the model.
        train_loader (DataLoader): The dataloader that contains the training samples.
        optimizer: The optimizer to use for training.
        epoch (int): The index of the current epoch.
        writer: For TensorBoard logging.

    Returns:
        loss_it_avg (list): The list of all the losses on each batch for the epoch.
    """
    model.train()
    device = args['device']
    writer = args['writer']
    loss_it = []
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    for it, batch in tqdm(enumerate(train_loader), desc="Epoch %s: " % (epoch+1), total=train_loader.__len__()):
        batch = {'anchor': batch['anchor'].to(device), 'positive': batch['positive'].to(device), 'negative' : batch['negative'].to(device), 'label': batch['label'].to(device)}
        optimizer.zero_grad()

        # perform training
        A, P, N = model(batch['anchor'], batch['positive'], batch['negative'])
        loss = triplet_loss(A, P, N)
        loss.backward()
        optimizer.step()

        # store loss history
        loss_it.append(loss.item())

    loss_it_avg = sum(loss_it)/len(loss_it)

    # print useful information about the training progress and scores on this training set's full pass
    print("Epoch %s/%s - %s : (%s %s)" % (colored(str(epoch+1), 'blue'),args['max_eps'] , colored('Training', 'blue'), colored('Average loss: ', 'cyan'), sum(loss_it)/len(loss_it)))

	# 🛑 add some metrics to keep with a label and the epoch index
    writer.add_scalar("Loss/train", loss_it_avg, epoch)

    return loss_it_avg


def train_contextual(args, model, train_loader, epoch):
    """
    Perform one epoch of training on the Sentence-EmCoBERT model.

    Args:
        args (str): The hyperparameters for the training.
        model: The model to train.
        device (str): The device where to send the inputs and the model.
        train_loader (DataLoader): The dataloader that contains the training samples.
        epoch (int): The index of the current epoch.

    Returns:
        ce_loss_avg (list): The list of all the cross-entropy losses on each batch for the epoch.
        triplet_loss_avg (list): The list of all the triplet losses on each batch for the epoch.
    """
    model.train()
    device = args['device']

    # set optimizers and loss
    optimizer_ce = optim.SGD(model.parameters(), lr = args['lr'], momentum=0.9)
    # optimizer_ce = optim.Adam(model.parameters(), lr = args['lr'])
    # optimizer_ce = optim.ASGD(model.parameters(), lr=args['lr'], lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0, foreach=None, maximize=False, differentiable=False)
    # optimizer_triplet = optim.Adam(model.parameters(), lr = args['lr'])
    optimizer_triplet = optim.AdamW(model.parameters(), lr=args['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    # set empty lists to track variables values
    ce_loss_it, triplet_loss_it = [], []
    all_utterances, all_labels = [], []
    encoder_trues, encoder_preds, triplet_trues, triplet_preds = [], [], [], []

    # prepare using only the TRE part to produce contextual utterances representations
    model.set_status_to_encoder()

    # train the encoder
    for it, batch in tqdm(enumerate(train_loader), desc="Epoch %s: " % (epoch+1), total=train_loader.__len__()):
        batch = {'embeddings' : batch['embedding'].to(device), 'labels' : batch['label'].to(device)}

        utterances_it = model(batch['embeddings'])
        utterances_it_cleaned = []

        # process and store the utterances and labels
        labels_list = batch['labels'].tolist()
        labels_without_pad = [labels[:labels.index(-1) if -1 in labels else len(labels)] for labels in labels_list]
        
        all_labels_pad_included = batch['labels'].flatten()
        for i in range(utterances_it.size()[0]):
            utterance = utterances_it[i]
            if all_labels_pad_included[i] != -1:
                utterances_it_cleaned.append(torch.tensor(utterance))

        utterances_it_final = torch.stack(utterances_it_cleaned)

        all_utterances.append(utterances_it_final)
        all_labels.append(torch.tensor(custom_flatten(labels_without_pad)))

    # prepare training with triplet loss
    train_loader_encoded = dataloader_from_encoder_output(args, 'train', all_utterances, all_labels)
    model.set_status_to_triplet()

    # train with triplet loss
    for it, batch in tqdm(enumerate(train_loader_encoded), desc="Epoch %s: " % (epoch+1), total=train_loader_encoded.__len__()):
        batch = {"anchor": batch["anchor"].to(device), "positive": batch["positive"].to(device), "negative": batch["negative"].to(device), "labels": batch["label"].to(device)}

        optimizer_ce.zero_grad()

        outputs = []
        # double for loop to process the examples in the same order that labels are sorted
        for i in range(len(batch["anchor"])):
            a, p, n = batch["anchor"][i], batch["positive"][i], batch["negative"][i]
            for embedding in [a, p, n]:
                classes_probas = model(embedding)
                outputs.append(classes_probas)

        # flatten at both batch and entry level
        outputs = torch.cat(outputs)

        labels = batch["labels"].flatten().long()
        weights = get_labels_weights(batch, device).to(device)
        ce_loss = nn.CrossEntropyLoss(weight=weights)
        loss = ce_loss(outputs, labels)
        loss.backward()
        optimizer_ce.step()

        ce_loss_it.append(loss.item())
        _, preds  = torch.max(outputs, 1)
        encoder_preds.append(preds)
        encoder_trues.append(labels)

        # apply triplet loss
        optimizer_triplet.zero_grad()

        # compute individual losses for evaluation
        labels = batch['labels'].tolist()
        for i in range(len(batch['anchor'])):
            loss_indiv = triplet_loss(batch['anchor'][i], batch['positive'][i], batch['negative'][i])
            true = labels[i][0] # anchor class
            triplet_trues.append(true)  
            if loss_indiv.item() <= 1.0 + 1e-4:
                pred = true # positive class
            else:
                pred = labels[i][2] # negative class
            triplet_preds.append(pred)

        # compute loss on the whole batch for backpropagation
        loss = triplet_loss(batch['anchor'], batch['positive'], batch['negative'])
        triplet_loss_it.append(loss.item())
        loss.requires_grad = True
        loss.backward()
        optimizer_triplet.step()

    # compute metrics to display (encoder)
    encoder_trues, encoder_preds = torch.cat(encoder_trues).flatten().tolist(), torch.cat(encoder_preds).flatten().tolist()
    ce_loss_avg = sum(ce_loss_it)/len(ce_loss_it)
    mcc = matthews_corrcoef(encoder_trues, encoder_preds)
    precision = precision_score(encoder_trues, encoder_preds, average='weighted', zero_division=0)
    recall = recall_score(encoder_trues, encoder_preds, average="weighted")
    f1score = f1_score(encoder_trues, encoder_preds, average='weighted')

    # affichage scores après CE : loss, MCC, Precision, Recall, F1
    print("Epoch %s/%s - %s : (%s %s) (%s %s) (%s %s) (%s %s) (%s %s)" % (colored(str(epoch+1), 'blue'), args['max_eps'] , colored('Tr. encoder', 'blue'), colored('Average CE loss: ', 'cyan'), ce_loss_avg, colored('MCC: ', 'cyan'), mcc, colored('Precision: ', 'cyan'), precision, colored('Recall: ', 'cyan'), recall, colored('F1: ', 'cyan'), f1score))

    # compute metrics to display
    triplet_loss_avg = sum(triplet_loss_it)/len(triplet_loss_it)
    mcc = matthews_corrcoef(triplet_trues, triplet_preds)
    micro_f1 = f1_score(encoder_trues, encoder_preds, average='micro') 
    macro_f1 = f1_score(encoder_trues, encoder_preds, average='macro')
    f1score = f1_score(triplet_trues, triplet_preds, average='weighted')
    print(colored(f"CE loss max: {max(ce_loss_it)}", 'green'))
    print("Epoch %s/%s - %s : (%s %s) (%s %s) (%s %s) (%s %s) (%s %s)" % (colored(str(epoch+1), 'blue'), args['max_eps'] , colored('Tr. triplet', 'blue'), colored('Average triplet loss: ', 'cyan'), triplet_loss_avg, colored('MCC: ', 'cyan'), mcc, colored('Macro F1: ', 'cyan'), macro_f1, colored('Micro F1: ', 'cyan'), micro_f1, colored('F1: ', 'cyan'), f1score))

	# 🛑 add some metrics to keep with a label and the epoch index
    # writer.add_scalar("Loss/train", loss_it_avg, epoch)

    return ce_loss_avg, triplet_loss_avg


def dataloader_from_encoder_output(args, target, utterances, labels):
    """
    Create a DataLoader object to be used in the training loop after the encoder training.

    From the utterance representations and the gold labels, this function creates a DataLoader object
    that can be used to perform the second part of the training using triplet loss.

    Args:
        args (dict): The params dictionary that contains, in particular, the train batch size.
        target (str): Either 'train', 'validation', or 'test', to indicate which dataset is being processed.
        utterances (list): A list of torch tensors that represent the utterances of the batch.
        labels (list): A list of lists of labels for each dialog.

    Returns:
        loader (DataLoader): The loader from the dataset mapping each learned utterance representation to its gold label.
    """
    utterances_processed = torch.cat(utterances)
    labels_processed = torch.cat(labels).flatten()

    all_encoded_dataset = {target:datasets.Dataset.from_dict({'vectors':utterances_processed.requires_grad_(), 'label':labels_processed})}

    try: 
        target in ['train', 'validation', 'test']
    except:
        print("Invalid target name. Accepted target names are 'train', 'validation' and 'test'.")

    # instantiate dataloader in consequence
    if target == 'train':
        loader = DataLoader(dataset=UtteranceVectorsEmotionDataset(all_encoded_dataset[target], args=args), batch_size=args['train_bsize'], shuffle=True, drop_last=True)
    else:
        loader = DataLoader(dataset=UtteranceVectorsEmotionDataset(all_encoded_dataset[target], args=args), batch_size=args['eval_bsize'], shuffle=True, drop_last=True)

    return loader


def test_isolated(target, model, loader):
    """
    Perform model evaluation on the validation or test set for one epoch, in the case of the isolated utterance model.

    Args:
        target (str): Indicates on which set the model is evaluated, for a coherent display.
        model: The model to evaluate.
        loader (DataLoader): The dataloader, either validation or test loader.
        device (str): The device where to send the inputs and the model.

    Returns:
        val_loss_avg (float): The average value of the loss on all the batches for one epoch.
        mcc (float): MCC score computed on the target dataset.
        micro_f1 (float): Micro F1 score computed on the target dataset.
        macro_f1 (float): Macro F1 score computed on the target dataset.
        f1score (float): Weighted F1 score computed on the target dataset.
        trues (list): All gold labels on the target dataset.
        preds (list): All predicted classes on the target dataset.
    """
    model.eval()
    loss_it = []
    preds, trues = [], []
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    device = args['device']

    for it, batch in tqdm(enumerate(loader), desc="%s: " % (target), total=loader.__len__()):

        with torch.no_grad():
            
            batch = {'anchor': batch['anchor'].to(device), 'positive': batch['positive'].to(device), 'negative' : batch['negative'].to(device), 'label': batch['label'].to(device)}

            A, P, N = model(batch['anchor'], batch['positive'], batch['negative'])
            loss = triplet_loss(A,P,N)
            loss_it.append(loss.item())

            # retrive true class and predicted class depending on the loss value
            labels = batch['label'].flatten().tolist()
            trues.append(labels[0]) # anchor class
            if loss.item() <= 1.0 + 1e-4:
                pred = labels[0] # positive class
            else:
                pred = labels[2] # negative class
            preds.append(pred)

    mcc = matthews_corrcoef(trues, preds)
    micro_f1 = f1_score(trues, preds, average='micro', labels=[1,2,3,4,5,6]) 
    macro_f1 = f1_score(trues, preds, average='macro', labels=[1,2,3,4,5,6])
    f1score = f1_score(trues, preds, average='weighted', labels=[1,2,3,4,5,6])

    val_loss_avg = sum(loss_it)/len(loss_it)

    print("%s : (%s %s) (%s %s) (%s %s) (%s %s) (%s %s)" % (colored(target, 'blue'), colored('Average triplet loss: ', 'cyan'), val_loss_avg, colored('MCC: ', 'cyan'), mcc, colored('Macro F1: ', 'cyan'), macro_f1, colored('Micro F1: ', 'cyan'), micro_f1, colored('F1: ', 'cyan'), f1score))

    return val_loss_avg, mcc, micro_f1, macro_f1, f1score, trues, preds


def test_contextual(target, model, loader):
    """
    Perform model evaluation on the validation or test set for one epoch, in the case of the Sentence-EmCoBERT model.

    Args:
        target (str): Indicates on which set the model is evaluated, for a coherent display.
        model: The model to evaluate.
        loader (DataLoader): The dataloader, either validation or test loader.
        device (str): The device where to send the inputs and the model.

    Returns:
        val_loss_avg (float): The average value of the loss on all the batches for one epoch.
        mcc (float): MCC score computed on the target dataset.
        micro_f1 (float): Micro F1 score computed on the target dataset.
        macro_f1 (float): Macro F1 score computed on the target dataset.
        f1score (float): Weighted F1 score computed on the target dataset.
        trues (list): All gold labels on the target dataset.
        preds (list): All predicted classes on the target dataset.
    """
    model.eval()
    model.set_status_to_encoder()
    loss_it = []
    preds, trues = [], []
    all_utterances, all_labels = [], []
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    device = args['device']

    with torch.no_grad():
        
        for it, batch in tqdm(enumerate(loader), desc="%s: " % (f"{target} encoding"), total=loader.__len__()):

            batch = {'embeddings' : batch['embedding'].to(device), 'labels' : batch['label'].to(device)}

            utterances_it = model(batch['embeddings'])
            utterances_it_cleaned = []

            # process and store the utterances and labels
            labels_list = batch['labels'].tolist()
            labels_without_pad = [labels[:labels.index(-1) if -1 in labels else len(labels)] for labels in labels_list]
            
            all_labels_pad_included = batch['labels'].flatten()
            for i in range(utterances_it.size()[0]):
                utterance = utterances_it[i]
                if all_labels_pad_included[i] != -1:
                    utterances_it_cleaned.append(torch.tensor(utterance))

            utterances_it_final = torch.stack(utterances_it_cleaned)

            all_utterances.append(utterances_it_final)
            all_labels.append(torch.tensor(custom_flatten(labels_without_pad)))

        # instantiate a dataloader with all utterances representations and associated labels
        loader_encoded = dataloader_from_encoder_output(args, target, all_utterances, all_labels)

        for it, batch in tqdm(enumerate(loader_encoded), desc="%s" % (f"{target} triplet"), total=loader_encoded.__len__()):
            
            batch = {'anchor': batch['anchor'].to(device), 'positive': batch['positive'].to(device), 'negative' : batch['negative'].to(device), 'labels': batch['label'].to(device)}

            # compute triplet loss
            loss = triplet_loss(batch['anchor'], batch['positive'], batch['negative'])
            loss_it.append(loss.item())

            # retrive true class and predicted class depending on the loss value
            labels = batch['labels'].flatten().tolist()
            trues.append(labels[0]) # anchor class
            if loss.item() <= 1.0 + 1e-4:
                pred = labels[0] # positive class
            else:
                pred = labels[2] # negative class
            preds.append(pred)

    mcc = matthews_corrcoef(trues, preds)
    micro_f1 = f1_score(trues, preds, average='micro', labels=[1,2,3,4,5,6])
    macro_f1 = f1_score(trues, preds, average='macro', labels=[1,2,3,4,5,6])
    f1score = f1_score(trues, preds, average='weighted', labels=[1,2,3,4,5,6])

    val_loss_avg = sum(loss_it)/len(loss_it)

    print("%s : (%s %s) (%s %s) (%s %s) (%s %s) (%s %s)" % (colored(target, 'blue'), colored('Average triplet loss: ', 'cyan'), val_loss_avg, colored('MCC: ', 'cyan'), mcc, colored('Macro F1: ', 'cyan'), macro_f1, colored('Micro F1: ', 'cyan'), micro_f1, colored('F1: ', 'cyan'), f1score))

    return val_loss_avg, mcc, micro_f1, macro_f1, f1score, trues, preds


def write_predictions(trues, preds, exp):
    """
    Retrieve the trained model and create a CSV file containing the detailed predictions facing the real labels.

    Args:
        trues (list): The gold labels from the DailyDialog test set.
        preds (list): The predicted labels from the model `model_name`.
        model_name (str): Name of the saved model on which inferences are made.
    """
    if not os.path.exists('../results/'):
        os.makedirs('../results/')  # create results dir if it doesn't exist

    with open(f'../results/predictions_{exp}.tsv', 'w', newline='') as f:
        write = csv.writer(f)

        # Write header
        write.writerow(['True Labels', 'Predicted Labels'])

        # Write data in columns
        for true, pred in zip(trues, preds):
            write.writerow([true, pred])


def run_epochs_isolated(model, args, optimizer, train_loader):
    """
    Perform the training and evaluate the model on the validation set for all epochs, in the case of the isolated utterance model.

    Args:
        model: The model to train and evaluate.
        args (dict): The hyperparameters for the training.
        optimizer: The optimizer to use for training.
        train_loader (DataLoader): The dataloader that contains training samples.
        experiment (str): The experiment name to name the logging folder.

    Returns:
        val_losses (list): A list that contains the validation loss values for each epoch.
    """
    val_losses = []
    early_stopping = EarlyStopping(tolerance=3, min_delta=0.1)
    writer = args['writer']

    for ep in range(args['max_eps']):
        # train the model on the train loader
        train_loss = train_isolated(args, model, train_loader, optimizer, ep)
        # infer on the validation loader
        val_loss_avg, mcc,  micro_f1, macro_f1, f1score, _, _ = test_isolated("validation", model, val_loader)
        # 🛑
        writer.add_scalar("Loss/val", val_loss_avg, ep)
        writer.add_scalar("MCC/val", mcc, ep)
        writer.add_scalar("MicroF1/val", micro_f1, ep)
        writer.add_scalar("MacroF1/val", macro_f1, ep)
        writer.add_scalar("F1/val", f1score, ep)
		# append the validation losses (good losses should normally go down)
        val_losses.append(val_loss_avg)
        
        # check early stopping
        early_stopping(train_loss, val_loss_avg)
        if early_stopping.early_stop:
            print(colored(f"Early stopping triggered. We are at epoch {ep+1}.", 'magenta'))
            torch.save(model.state_dict(), '../models/final_model_isolated.pt')
            break

	# 🛑 flush to perform all remaining operations
    writer.flush()

    return val_losses


def run_epochs_contextual(model, args, sm, warmup_steps):
    """
    Perform the training and evaluate the model on the validation set for all epochs, in the case of the Sentence-EmCoBERT model.

    Args:
        model: The model to train and evaluate.
        args (dict): The hyperparameters for the training.
        device (str): The device where to send the inputs and the model.
        writer: For TensorBoard logging.
        warmup_steps (int): The number of desired warmup steps.

    Returns:
        val_losses (list): A list that contains the validation loss values for each epoch.
    """
    val_losses = []
    lr = args['lr']
    writer = args['writer']
    best_model = model
    biggest_mcc = 0

    # training warmup steps
    if warmup_steps > 0:
        print(colored(f"\nWarmup training steps", 'magenta'))
        args.update({"lr":0.00001})
        for ep in range(warmup_steps):
            ce_warm_loss, triplet_warm_loss = train_contextual(args, model, train_loader, ep)
            # 🛑
            writer.add_scalar("CE_warm_loss/train", ce_warm_loss, ep)
            writer.add_scalar("Triplet_warm_loss/train", triplet_warm_loss, ep)

    # main training epochs
    print(colored(f"\nStart of actual training epochs", 'magenta'))
    args.update({"lr":lr}) # reset lr arg to initial value
    for ep in range(args['max_eps']):
        # training
        ce_loss, triplet_loss = train_contextual(args, model, train_loader, ep)
        # 🛑
        writer.add_scalar("CE_loss/train", ce_loss, ep)
        writer.add_scalar("Triplet_loss/train", triplet_loss, ep)
        # inference
        val_loss_avg, mcc, micro_f1, macro_f1, f1score, _, _ = test_contextual("Validation", model, val_loader)
        # 🛑
        writer.add_scalar("Loss/val", val_loss_avg, ep)
        writer.add_scalar("MCC/val", mcc, ep)
        writer.add_scalar("MicroF1/val", micro_f1, ep)
        writer.add_scalar("MacroF1/val", macro_f1, ep)
        writer.add_scalar("F1/val", f1score, ep)
		# append the validation losses (good losses should normally go down)
        val_losses.append(val_loss_avg)

        # best model saving strategy
        if mcc > biggest_mcc:
            biggest_mcc = mcc
            best_model = model
    
    # save best model to pickle file
    torch.save(best_model.state_dict(), f'../models/final_model_{sm}.pt')

	# 🛑 flush to perform all remaining operations
    writer.flush()

    return val_losses


def compute_standard_deviations_on_test(target, model, loader, test_function, exp, episodes=10):
    """
    Run inference on DailyDialog test for the intended number of episodes (default=10) and compute descriptive statistics on classification metrics.

    Args:
        target (str): Either 'validation' or 'test'. Intended to be consistent with the provided loader.
        model: The model to evaluate.
        device (str): The device where to send the inputs and the model.
        loader (DataLoader): The dataloader that contains training samples.
        test_function: The function to use to perform inference. Depends on the model (BERT or Sentence-BERT based).
        exp (str): Either 'isolated' or 'contextual', to put in the output file name.
        episodes (int): The number of times the test set should be inferred on (default=10).

    Returns:
        results (dict): The means and the standard deviations of all computed classification metrics (loss, MCC, micro F1, macro F1, weighted F1).
    """
    loss_list, mcc_list, micro_list, macro_list, f1_list = [], [], [], [], []

    for _ in range(episodes):
        val_loss_avg, mcc, microf1, macrof1, f1score, trues, preds = test_function(target, model, loader)
        loss_list.append(val_loss_avg)
        mcc_list.append(mcc)
        micro_list.append(microf1)
        macro_list.append(macrof1)
        f1_list.append(f1score)

    # store predicted labels from last run along with gols labels in a csv file
    write_predictions(trues, preds, exp)
    
    std_loss, mean_loss = np.std(loss_list), np.mean(loss_list)
    std_mcc, mean_mcc = np.std(mcc_list), np.mean(mcc_list)
    std_micro, mean_micro  = np.std(micro_list), np.mean(micro_list)
    std_macro, mean_macro = np.std(macro_list), np.mean(macro_list)
    std_f1, mean_f1  = np.std(f1_list), np.mean(f1_list)

    results = {'mean' : [mean_loss, mean_mcc, mean_micro, mean_macro, mean_f1], 'std' : [std_loss, std_mcc, std_micro, std_macro, std_f1]}

    return results


def run_exp_contextual(lr_list, warmup_steps, pretrained, n_layers=1, episodes=10):
    """
    Perform the end-to-end experiment using the Sentence-EmCoBERT model only.

    Args:
        lr_list (list): The learning rates to use sequentially for the experiments.
        warmup_steps (int): The number of warmup steps. 0 means no warmup.
        pretrained (str): The name of the pretrained sentence transformer model to use in the experiment.
        n_layers (int): Number of transformer encoder layers to contextualize the input representation.
        episodes (int): The number of times the test set should be inferred on by the trained model.
    """
    max_lengths = {'all-MiniLM-L6-v2':384, 'all-roberta-large-v1':1024, 'all-mpnet-base-v2':768}

    for l_rate in lr_list:

        args.update({"lr": l_rate})

        # Set the summary writer with a specific name
        writer = SummaryWriter(f"{experiment}/{get_datetime()}", comment=args2filename(args))

        # Set device and model
        device = args['device']
        max_length = max_lengths[pretrained]
        # model = SentenceEmCoBERT(args, device, pretrained, max_len=max_length, n_layers=n_layers)
        model = SentenceEmCoBERT(args, device, n_layers, max_len=max_length, sm=pretrained)
        model.to(device)

        # Perform training and testing + log scores
        val_losses = run_epochs_contextual(model, args, pretrained, warmup_steps)
        means_and_std = compute_standard_deviations_on_test('test', model, test_loader, test_contextual, 'contextual', episodes)
        means = means_and_std['mean']
        std = means_and_std['std']
        
        test_metrics = {'hparam/loss':      round(means[0], 4),
                        'hparam/mcc':     round(means[1], 4),
                        'hparam/precision':round(means[2], 4),
                        'hparam/recall':   round(means[3], 4),
                        'hparam/f1':       round(means[4], 4)}
        
        print(colored("End of experiment - variances of the metrics", 'yellow'))
        print(f'Loss: {std[0]}\t MCC: {std[1]}\t Precision: {std[2]}\t Recall {std[3]}\t F1: {std[4]}')
        # 🛑 save the hparams
        writer.add_hparams(test_metrics)
        writer.flush()

    # 🛑 close the writer
    writer.close()


def run_exp_isolated(lr_list, episodes=10):
    """
    Perform the end-to-end experiment using the isolated utterance model only.

    Args:
        lr_list (list): The learning rates to use sequentially for the experiments.
        episodes (int): The number of times the test set should be inferred on by the trained model.
    """
    arguments = parser.parse_args()
    experiment = arguments.exp

    for l_rate in lr_list:
        print(f"Current learning rate: {l_rate}")
        args.update({"lr": l_rate})

        # Set the summary writer with a specific name
        writer = SummaryWriter(f"{experiment}/{get_datetime()}", comment=args2filename(args))

        # Set device and model
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        model = SiameseNetworkLSTM(n_layers=5, input_dim=20, hidden_dim=300)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr = args['lr'])

        # Perform training and testing + log scores
        val_losses = run_epochs_isolated(model, args, optimizer, train_loader)
        means_and_std = compute_standard_deviations_on_test('test', model, test_loader, test_isolated, 'isolated', episodes)
        means = means_and_std['mean']
        std = means_and_std['std']

        test_metrics = {'hparam/loss':      round(means[0], 4),
                        'hparam/mcc':     round(means[1], 4),
                        'hparam/precision':round(means[2], 4),
                        'hparam/recall':   round(means[3], 4),
                        'hparam/f1':       round(means[4], 4)}
        
        print(colored("End of experiment - variances of the metrics", 'yellow'))
        print(f'Loss: {std[0]}\t MCC: {std[1]}\t Precision: {std[2]}\t Recall {std[3]}\t F1: {std[4]}')
        # 🛑 save the hparams
        writer.add_hparams(args, test_metrics)
        writer.flush()

    # 🛑 close the writer
    writer.close()


def run_quali_contextual(pretrained, file_path, dataset):
    """
    Run a qualitative experiment on contextual utterances using the Sentence-EmCoBERT model.

    It evaluates about 500 examples from the test set using the triplet loss principle to compute and display predictions
    compared to actual labels. The output is stored in a txt file called `output.txt` and follows a specific layout:
    for each example, it provides the dialog, the considered utterance, and the true/predicted labels. Correct predictions
    are marked with ✅ whereas incorrect ones are marked with ❌.

    Args:
        device (str): Choose the device on which you want to apply the model.
        max_length (int): The utterance padding length associated with the desired Sentence-EmCoBERT model.
    """
    dailydialog = datasets.load_dataset("daily_dialog")

    # model and loss setup
    device = args['device']
    model = SentenceEmCoBERT({}, device, 1, pretrained)
    model.set_status_to_encoder # apply the encoder to the test samples
    model.load_state_dict(torch.load(f'../models/final_model_{pretrained}.pt'))
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    indexes_tripletable = find_tripletable_indexes(dailydialog)

    for index in indexes_tripletable:

        sample = dataset['test'][index]
        sample['embedding'] = torch.tensor(sample['embedding']).unsqueeze(0)
        text = dailydialog['test'][index]['dialog']
        with open(file_path, "a") as f:
            print(50*"=", file=f)
            print(20*"=" + f"DIALOG #{index}" + 20*"=", file=f)
            print(50*"=", file=f)
        for utt in text:
            with open(file_path, "a") as f:
                print("- ", utt, file=f)

        output = model(sample['embedding'].to(device))
        A, P, N, index_A, index_P, index_N = find_triplet_indexes(dailydialog['test'][index]['emotion'])
        loss = triplet_loss(output[index_A], output[index_P], output[index_N])
        real = map_index_to_emotions(A)
        neg = map_index_to_emotions(N)
        if loss.item() <=1 + 1e-4:
            with open(file_path, "a") as f:
                print("", file=f)
                print("Utterance: ", dailydialog['test'][index]['dialog'][index_P], "\nEmotion:", real, "Prediction:", real, "✅", file=f)
        else:
            with open(file_path, "a") as f:
                print("", file=f)
                print("Utterance: ", dailydialog['test'][index]['dialog'][index_P], "\nEmotion:", real, "Prediction:", neg, "❌",file=f)  
        with open(file_path, "a") as f:
            print("",file=f)


def run_quali_isolated(file_path, device='cuda'):
    """
    Run a partial qualitative experiment on static utterances.

    It aims at checking specific examples identified in the contextual model qualitative analysis. The output layout
    is similar to the one generated by `run_quali_context()` and is stored in a txt file called `output_static.txt`.
    For readability and easier probing, correct predictions are marked with ✅ whereas incorrect ones are marked with ❌.

    The considered examples in the experiment are the following:
        - SAE = Still An Emotion: Wrong predictions where ground truth label was 'no emotion', but the utterance still
          conveys a feeling that is not described by DailyDialog labels.
        - CDH = Context Did Help: Correct predictions where the dialog context seems to have helped produce the right result.
        - CSH = Context Should Help: Incorrect prediction where the dialog context should have helped in producing the right emotion.

    Args:
        device (str): Default is 'cuda'. Choose the device on which you want to apply the model.
    """
    dailydialog = datasets.load_dataset("daily_dialog")

    # model and loss setup
    device = device
    model = SiameseNetworkLSTM(n_layers=5, input_dim=20, hidden_dim=300)
    model.load_state_dict(torch.load("../models/final_model_isolated.pt"))
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    # store indexes of relevant examples
    sae = [197, 304, 307, 390, 547]
    cdh = [125, 134, 143, 763] 
    csh = [108, 176, 291, 294, 354, 651] 
    sae_arr, cdh_arr, csh_arr = get_quali_static_arrays(dailydialog, sae, cdh, csh)

    for i in range(len(csh)):
        index = csh[i]
        vectors = [x['vectors'] for x in utterances_dataset['test'] if x['index']==index]
        text = dailydialog['test'][index]['dialog']
        with open(file_path, "a") as f:
            print(50*"=", file=f)
            print(20*"=" + f"DIALOG #{index}" + 20*"=", file=f)
            print(50*"=", file=f)
        for utt in text:
            with open(file_path, "a") as f:
                print("- ", utt, file=f)

        (A, P, N, index_A, index_P, index_N) = csh_arr[i]
        A_proc, P_proc, N_proc = model(torch.tensor(vectors[index_A]).unsqueeze(0).to(device), torch.tensor(vectors[index_P]).unsqueeze(0).to(device), torch.tensor(vectors[index_N]).unsqueeze(0).to(device))
        print("Model output ok.")
        loss = triplet_loss(A_proc, P_proc, N_proc)
        real = map_index_to_emotions(A)
        neg = map_index_to_emotions(N)
        if loss.item() <=1 + 1e-4:
            with open(file_path, "a") as f:
                print("", file=f)
                print("Utterance: ", dailydialog['test'][index]['dialog'][index_P], "\nEmotion:", real, "Prediction:", real, "✅", file=f)
        else:
            with open(file_path, "a") as f:
                print("", file=f)
                print("Utterance: ", dailydialog['test'][index]['dialog'][index_P], "\nEmotion:", real, "Prediction:", neg, "❌",file=f)  
        with open(file_path, "a") as f:
            print("",file=f)


if __name__ == "__main__":

    # Instantiate parser and retrieve arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Required argument. Choose between isolated (i), contextual (c) or sentence (s) utterance model", type=str) 
    parser.add_argument("-t", "--training", help="Run training and evaluation for the selected model", action='store_true') 
    parser.add_argument("-q", "--qualitative", help="Run qualitative experiment using the model associated to the selected utterance representation", action='store_true') 
    parser.add_argument("-e", "--exp", type=str, help="give a name to the experiment folder that stores Tensorflow logs", default="unnamed_experiment")
    parser.add_argument("-p", "--pretrained", type=str, help="for Sentence BERT, give the pretrained model to use. Use aliases among 'mpnet', 'roberta' and 'minilm'. Use 'all' to train on the three pretrained models one at a time.", default='all')
    parser.add_argument("-c", "--cpu", help="force CPU", action="store_true")

    arguments = parser.parse_args() 
    experiment = "../experiments/" + arguments.exp # set the path to store the logs
    pretrained_mapping = {'minilm':'all-MiniLM-L6-v2', 'mpnet':'all-mpnet-base-v2', 'roberta':'all-roberta-large-v1'}

    # Static utterance model training
    if arguments.model == 'i' and arguments.training:
        print("Selected experiment: ", colored('static utterance model training', 'green'))
        utterances_dataset = load_dataset("utterances_vectors")
        args, train_loader, val_loader, test_loader = get_args_and_dataloaders(utterances_dataset, UtteranceVectorsEmotionDataset)
        lr_list = [0.0005]
        # Set the summary writer with a specific name
        args.update({'max_eps':10, 'device':activate_gpu(force_cpu=arguments.cpu)})
        writer = SummaryWriter(f"{experiment}/{get_datetime()}", comment=args2filename(args))
        args.update({'writer':writer, 'experiment':experiment})

        run_exp_isolated(lr_list)

    # Contextual sentence BERT-based utterance model training
    if arguments.model == 's' and arguments.training:
        # check for the desired pretrained models (either all or one in particular)
        selected_pretrained = pretrained_mapping.values() if arguments.pretrained == 'all' else pretrained_mapping[arguments.pretrained]
        for pretrained in selected_pretrained:
            print("Selected experiment: ", colored('contextual Sentence BERT-based utterance model training', 'green'))
            sentences_dataset = load_dataset(f'sentences_processed_{pretrained}')
            print(f"Selected pretrained model: {pretrained}")
            args, train_loader, val_loader, test_loader = get_args_and_dataloaders_balanced(sentences_dataset, SentenceEmotionDatasetBERT, spreading=False)
            lr_list = [0.001, 0.0005]
            max_eps = [10]
            warmup_steps = 0
            n_layers = 1

            for lr in lr_list:
                print(f"Learning rate: {lr}")
                # Set the summary writer with a specific name
                args.update({'max_eps':max_eps[0], 'device':activate_gpu(force_cpu=arguments.cpu), 'lr':lr})
                writer = SummaryWriter(f"{experiment}/{get_datetime()}", comment=args2filename(args))
                args.update({'writer':writer, 'experiment':experiment})
                
                run_exp_contextual(lr_list, warmup_steps, pretrained, n_layers=n_layers)

    # Qualitative insights on static utterance model predictions
    if arguments.model == 'i' and arguments.qualitative:
        args = {'device':activate_gpu(force_cpu=arguments.cpu)}
        print("Selected experiment: ", colored('qualitative insights on static utterance model predictions', 'green'))
        run_quali_isolated(file_path="./outputs/quali_static.txt")

    # Qualitative insights on Sentence BERT contextual utterance model predictions
    if arguments.model == 's' and arguments.qualitative:
        print("Selected experiment: ", colored('qualitative insights on Sentence BERT contextual utterance model predictions', 'green'))
        args = {'device':activate_gpu(force_cpu=arguments.cpu)}
        # check for the desired pretrained models (either all or one in particular)
        selected_pretrained = pretrained_mapping.values() if arguments.pretrained == 'all' else pretrained_mapping[arguments.pretrained]
        for pretrained in selected_pretrained:
            print(f"Selected pretrained model: {pretrained}")
            sentences_dataset = load_dataset(f'sentences_processed_{pretrained}')
            run_quali_contextual(pretrained, "./outputs/quali_context_sentence.txt", sentences_dataset)