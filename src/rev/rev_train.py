"""
Generative model to predict any of RoT/action, situation, and attributes.
fine-tuning the encoder-decoder T5 model.
"""
import os
import torch
import logging
import argparse

from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss

from utils import init_model, load_data_ecqa, load_data_cose, load_data_esnli, load_data_quartz
from generative import evaluate, train, set_seed, load_and_cache_examples


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

load_data_func = {
    'ECQA': load_data_ecqa,
    'COSE': load_data_cose,
    'ESNLI': load_data_esnli,
    'QUARTZ': load_data_quartz
}
special_toks = {
    'ECQA': ["[question]", "[choice]", "[answer]", "[rationale]", "<eos>"],
    'COSE': ["[question]", "[choice]", "[answer]", "[rationale]", "<eos>"],
    'ESNLI': ["[premise]", "[hypothesis]", "[answer]", "[rationale]", "<eos>"],
    'QUARTZ': ["[question]", "[choice]", "[answer]", "[rationale]", "<eos>"]
}


class EncoderDecoderTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):
      logger.info("Converting to token IDs")
      examples = load_data_func[args.task](args, file_path, data_type=args.data_type) #load_data(file_path)
      logger.info(examples[:5])
      
      # Add prefix to the output so we can predict the first real token in the decoder
      inputs = [
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ex[0]))
            for ex in examples
      ]

      outputs = [
            [inputs[i][-1]]
            + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ex[1]))
            for i, ex in enumerate(examples)
      ]

      # Pad
      max_input_length = min(
            args.max_input_length, max([len(ex) for ex in inputs])
      )
      max_output_length = min(
            args.max_output_length, max([len(ex) for ex in outputs])
      )

      input_lengths = [min(len(ex), max_input_length) for ex in inputs]
      output_lengths = [min(len(ex), max_output_length) for ex in outputs]
      inputs = [tokenizer.encode(
            ex, add_special_tokens=False, max_length=max_input_length, padding='max_length', truncation=True)
            for ex in inputs]
      
      outputs = [tokenizer.encode(
            ex, add_special_tokens=False, max_length=max_output_length, padding='max_length', truncation=True)
            for ex in outputs]
      self.examples = {
            "inputs": inputs,
            "outputs": outputs,
            "input_lengths": input_lengths,
            "output_lengths": output_lengths,
      }

    def __len__(self):
        return len(self.examples["input_lengths"])

    def __getitem__(self, item):
        inputs = torch.tensor(self.examples["inputs"][item])
        outputs = torch.tensor(self.examples["outputs"][item])

        max_length = inputs.shape[0]
        input_lengths = self.examples["input_lengths"][item]
        input_mask = torch.tensor([1] * input_lengths + [0] * (max_length - input_lengths))

        max_length = outputs.shape[0]
        output_lengths = self.examples["output_lengths"][item]
        output_mask = torch.tensor([1] * output_lengths + [0] * (max_length - output_lengths))
        
        return {
            "inputs": inputs,
            "input_mask": input_mask,
            "outputs": outputs,
            "output_mask": output_mask,
        }


def get_loss(args, batch, model):
    """
    Compute this batch loss
    """
    input_ids = batch["inputs"].to(args.device)
    input_mask = batch["input_mask"].to(args.device)
    target_ids = batch["outputs"].to(args.device)
    output_mask = batch["output_mask"].to(args.device)
    decoder_input_ids = target_ids[:, :-1].contiguous()

    # We don't send labels to model.forward because we want to compute per token loss
    lm_logits = model(
        input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids, use_cache=False
    )[0]
    batch_size, max_length, vocab_size = lm_logits.shape

    # Compute loss for each instance and each token
    loss_fct = CrossEntropyLoss(reduction="none")
    lm_labels = target_ids[:, 1:].clone().contiguous()
    lm_labels[target_ids[:, 1:] == args.pad_token_id] = -100
    loss = loss_fct(lm_logits.view(-1, vocab_size), lm_labels.view(-1)).view(
        batch_size, max_length
    )

    # Only consider non padded tokens
    loss_mask = output_mask[..., :-1].contiguous()
    loss = torch.mul(loss_mask, loss)  # [batch_size, max_length]
    return loss


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        required=True,
        help="Out directory for checkpoints.",
    )

    # Other parameters
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--eval_during_train",
        action="store_true",
        help="Evaluate at each train logging step.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Steps before backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-6,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1000,
        help="Log every X updates steps (default after each epoch).",
    )
    parser.add_argument(
        "--max_input_length",
        default=75,
        type=int,
        help="Maximum input event length in words.",
    )
    parser.add_argument(
        "--max_output_length",
        default=50,
        type=int,
        help="Maximum output event length in words.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: total number of training steps to perform.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bart-large",
        type=str,
        help="LM checkpoint for initialization.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=2.0,
        type=float,
        help="Number of training epochs to perform.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached data."
    )
    parser.add_argument(
        "--overwrite_out_dir",
        action="store_true",
        help="Overwrite the output directory.",
    )
    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue training from the last checkpoint.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=-1,
        help="Save checkpoint every X updates steps (default after each epoch).",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )
    parser.add_argument(
        "--train_batch_size", default=8, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--train_path", default='train.csv', type=str, help="train dataset path."
    )
    parser.add_argument(
        "--val_path", default='dev.csv', type=str, help="val dataset path."
    )
    parser.add_argument(
        "--test_path", default='test.csv', type=str, help="test dataset path."
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--task",
        default="delta-nli",
        type=str,
        help="what is the task? ECQA , etc.",
    )
    parser.add_argument(
        "--task_model",
        default="t5-large",
        type=str,
        help="task model name",
    )
    parser.add_argument(
        "--data_type",
        default="regular",
        type=str,
        help="temp: template, regular: regular",
    )
    parser.add_argument(
        '--seednum', 
        action='store_true', 
        help='set seed num',
    )
    args = parser.parse_args()

    # for debug
    current_path = os.path.dirname(os.path.abspath(__file__))
    args.out_dir = os.path.join(current_path, args.out_dir)
    if args.seednum:
        args.out_dir = args.out_dir+'_seed_'+str(args.seed)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
        print("Directory '% s' created" % args.out_dir)
    
    if (
        os.path.exists(args.out_dir)
        and len(os.listdir(args.out_dir)) > 1
        and args.do_train
        and not args.overwrite_out_dir
        and not args.continue_training
    ):
        raise ValueError(
            f"Output directory {args.out_dir} already exists and is not empty. "
            f"Use --overwrite_out_dir or --continue_training."
        )

    # Setup device
    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )

    handlers = [
            logging.FileHandler(os.path.join(args.out_dir, "logger.log")),
            logging.StreamHandler(),
        ]
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )
    logger.info("Save path: %s" % args.out_dir)

    # load data
    if args.data_type == 'regular':
      args.data_path = os.path.join(current_path, '../', 'data', args.task)
      args.data_path = os.path.normpath(args.data_path)
      args.train_file = os.path.join(args.data_path, args.train_path)
      args.val_file = os.path.join(args.data_path, args.val_path)
      args.test_file = os.path.join(args.data_path, args.test_path)
    elif args.data_type == 'temp':
      args.train_file  = os.path.join(current_path, '../', 'templated_rationales', args.task, 'train.jsonl.predictions')
      args.train_file = os.path.normpath(args.train_file)
      args.val_file  = os.path.join(current_path, '../', 'templated_rationales', args.task, 'dev.jsonl.predictions')
      args.val_file = os.path.normpath(args.val_file)
      args.test_file  = os.path.join(current_path, '../', 'templated_rationales', args.task, 'test.jsonl.predictions')
      args.test_file = os.path.normpath(args.test_file)

    # Set seed
    set_seed(args)

    # Load the models
    if args.continue_training:
        args.model_name_or_path = args.out_dir
    # Delete the current results file
    else:
        eval_results_file = os.path.join(args.out_dir, "eval_results.txt")
        if os.path.exists(eval_results_file):
            os.remove(eval_results_file)

    args.device = "cpu"
    tokenizer, model = init_model(
        args.model_name_or_path, device=args.device, do_lower_case=args.do_lower_case
    )

    args.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Pad token ID: {args.pad_token_id}")
    args.block_size = tokenizer.max_len_single_sentence
    logger.info(f"Training/evaluation parameters {args}")

    # Add special tokens (if loading a model before fine-tuning)
    if args.do_train and not args.continue_training:
        special_tokens = special_toks[args.task]
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "<eos>"
        tokenizer.add_tokens(special_tokens)
        if 'gpt' in args.model_name_or_path:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))

    args.pad_token_id = tokenizer.pad_token_id

    # resize_token_embeddings for Bart doesn't work if the model is already on the device
    args.device = device
    model.to(args.device)

    # Training
    if args.do_train:
        if 'gpt' in args.model_name_or_path:
            train_dataset = load_and_cache_examples(
                args.train_file,
                args,
                tokenizer
            )
        else:
            train_dataset = EncoderDecoderTextDataset(
                tokenizer,
                args,
                file_path=args.train_file,
                block_size=args.block_size,
            )
        if args.do_eval or args.eval_during_training:
            if 'gpt' in args.model_name_or_path:
                eval_dataset = load_and_cache_examples(
                args.val_file, args, tokenizer)
            else:
                eval_dataset = EncoderDecoderTextDataset(
                tokenizer, args, file_path=args.val_file, block_size=args.block_size)

        if 'gpt' in args.model_name_or_path:
            global_step, tr_loss = train(
            args,
            train_dataset,
            model,
            tokenizer,
            eval_dataset=eval_dataset,
            )
        else:
            global_step, tr_loss = train(
                args,
                train_dataset,
                model,
                tokenizer,
                loss_fnc=get_loss,
                eval_dataset=eval_dataset,
            )

        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint = args.out_dir
        logger.info(f"Evaluate the following checkpoint: {checkpoint}")
        prefix = (
            checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        )
        tokenizer, model = init_model(
            checkpoint, device=args.device, do_lower_case=args.do_lower_case
        )

        model.to(args.device)
        if 'gpt' in args.model_name_or_path:
            eval_dataset = load_and_cache_examples(
                args.val_file, args, tokenizer)
            result = evaluate(eval_dataset, args, model, prefix=prefix)
        else:
            eval_dataset = EncoderDecoderTextDataset(
                tokenizer, args, file_path=args.val_file, block_size=args.block_size)
            result = evaluate(eval_dataset, args, model, prefix=prefix, loss_fnc=get_loss)
        results.update(result)

    return results


if __name__ == "__main__":
    main()
