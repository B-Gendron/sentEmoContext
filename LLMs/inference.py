from transformers import AutoTokenizer
import transformers
import torch
from utils import *
import argparse
from termcolor import colored


def run_inference(model, episodes=10):
    """
    Run inference using a pre-trained language model on dailydialog test set for the desired number of episoddes.

    Args:
        model: the huggingface path to pre-trained language model
        test_set: the original dailydialog test set
        episodes (int): the number of times the test set should be inferred on to compute statistics on the scores. Default=10
    """
    # get the data ( automatically loaded in utils.py)
    test_set = dailydialog['test']

    # set the tokenizer corresponding to the model
    tokenizer = AutoTokenizer.from_pretrained(model)

    # set the inference pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device="cuda",
    )

    # perform predictions on dailydialog test set
    all_preds = []
    for _ in range(episodes):
        print("Start inference")
        preds = []
        for i in range(len(test_set)):
            sequences = pipeline(
                format_prompt_last_utterance_falcon('test', i),
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=2000,
            )

            for seq in sequences:
                pred = map_emotion_to_index(predicted_emotion(seq['generated_text']))
            preds.append(pred)

        # update the vector containing all the predictions
        all_preds.append(preds)
    
    # build the gold labels vector
    all_trues = [[test_set[i]['emotion'][len(test_set[i]['emotion'])-1] for i in range(len(test_set))] for _ in range(10)]
    # log the scores
    results = compute_metrics_and_variance(all_trues, all_preds)
    store_classification_metrics(results, model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help="Name of the LLM on HuggingFace. This has to be a valid HF path. Default is 'meta-llama/Llama-2-7b-chat-hf'", default='meta-llama/Llama-2-7b-chat-hf')
    args = parser.parse_args()
    model = args.model

    print(colored(f'Selected model: {model}', 'green'))

    run_inference(model)