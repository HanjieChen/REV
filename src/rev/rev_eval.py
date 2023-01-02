"""
Adapted from https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
"""
import re
import json
import tqdm
import torch
import logging
import argparse
import os
import torch.nn as nn

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

from utils import init_model, load_data_ecqa, load_data_cose, load_data_esnli, load_data_quartz


load_data_func = {
    'ECQA': load_data_ecqa,
    'COSE': load_data_cose,
    'ESNLI': load_data_esnli,
    'QUARTZ': load_data_quartz,
}
special_toks = {
    'ECQA': ["[question]", "[choice]", "[answer]", "[rationale]", "<eos>"],
    'COSE': ["[question]", "[choice]", "[answer]", "[rationale]", "<eos>"],
    'ESNLI': ["[premise]", "[hypothesis]", "[answer]", "[rationale]", "<eos>"],
    'QUARTZ': ["[question]", "[choice]", "[answer]", "[rationale]", "<eos>"],
}


def main() -> None:
    """
    Generate intensifiers and attenuators
    """
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument(
        "--model_name",
        default="t5-large",
        type=str,
        help="LM checkpoint for initialization.",
    )

    # Optional
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
        "--max_length", default=120, type=int, required=False, help="Maximum text length"
    )
    parser.add_argument(
        "--min_length", default=5, type=int, required=False, help="Maximum text length"
    )
    parser.add_argument(
        "--k", default=0, type=int, required=False, help="k for top k sampling"
    )
    parser.add_argument(
        "--p", default=0, type=float, required=False, help="p for nucleus sampling"
    )
    parser.add_argument(
        "--beams", default=0, type=int, required=False, help="beams for beam search"
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        required=False,
        help="temperature for sampling",
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--task",
        default="delta-nli",
        type=str,
        help="what is the task? delta-nli or scarecrow , etc.",
    )
    parser.add_argument(
        "--task_model",
        default="t5-large",
        type=str,
        help="task model name",
    )
    parser.add_argument(
        "--test_type",
        default="gen",
        type=str,
        help="gold: gold rationales, gen: generative rationales",
    )
    parser.add_argument(
        "--out_type",
        default="YR",
        type=str,
        help="Output type: 'YR' label and rationale, 'Y' label only, 'R' rationale only",
    )
    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="train, val, test",
    )
    parser.add_argument(
        "--data_type",
        default="regular",
        type=str,
        help="temp: template, regular: regular",
    )
    args = parser.parse_args()
    logger.debug(args)

    if (
        (args.k == args.p == args.beams == 0)
        or (args.k != 0 and args.p != 0)
        or (args.beams != 0 and args.p != 0)
        or (args.beams != 0 and args.k != 0)
    ):
        raise ValueError(
            "Exactly one of p, k, and beams should be set to a non-zero value."
        )

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    logger.debug(f"Initializing {args.device}")
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    if args.split == 'train':
            split_type = 'train'
    elif args.split == 'val':
            split_type = 'dev'
    elif args.split == 'test':
            split_type = 'test'


    # compute the vi of rationales
    rat_outputs = compute_vi(args, current_path, data_type='regular', device=device)
    
    # compute the vi of baselines
    base_outputs = compute_vi(args, current_path, data_type='temp', device=device)

    rat_vi_list = []
    base_vi_list = []
    for rat_output, base_output in zip(rat_outputs, base_outputs):
        rat_vi_list.append(rat_output[3])
        base_vi_list.append(base_output[3])

    # compute cvi
    cvi = sum(rat_vi_list) / len(rat_vi_list) - sum(base_vi_list) / len(base_vi_list)

    print('cvi:{}'.format(cvi))


def compute_vi(args, current_path, data_type, device):
    if data_type == 'regular':
        args.data_path = os.path.join(current_path, '../path_to_gold_or_generated_rationale')
        args.in_file = os.path.join(args.data_path, args.split+'_'+'filename.jsonl')
    elif data_type == 'temp':
        args.data_path = os.path.join(current_path, '../path_to_template_rationale')
        args.in_file = os.path.join(args.data_path, args.split+'_'+'filename.jsonl')
    
    args.model_name_or_path = os.path.join(current_path, 'output', args.task+'_'+'regular'+'-'+args.model_name)
    args.out_file = os.path.join(current_path, 'output', 'output_filename.jsonl')

    tokenizer, model = init_model(args.model_name_or_path, device)
    examples = load_data_func[args.task](args, args.in_file, data_type=data_type, test_type=args.test_type, out_type=args.out_type)

    generate = (
        generate_conditional
        if "t5" in args.model_name_or_path or "bart" in args.model_name_or_path
        else generate_regular
    )
 

    output_list = []
    with open(args.out_file, "w") as f_out:
        for input, output in tqdm.tqdm(examples):
            try:
                info, preds = generate(
                    tokenizer,
                    model,
                    args,
                    input,
                    output,
                    device,
                )

            except Exception as exp:
                logger.info(exp)
                preds = []

            f_out.write(
                json.dumps({"input": input, "label": output, "predictions": preds, "info": info.item()})
                + "\n"
            )
            output_list.append((input, output, preds, info.item()))
    return output_list


def generate_conditional(tokenizer, model, args, input, output, device):
    """
    Generate a sequence with models like Bart and T5
    """
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input))
    decoder_start_token_id = input_ids[-1]
    input_ids = torch.tensor([input_ids]).to(device)
    max_length = args.max_length
    min_length = args.min_length

    outputs = model.generate(
        input_ids,
        do_sample=args.beams == 0,
        max_length=max_length,
        min_length=min_length,
        temperature=args.temperature,
        top_p=args.p if args.p > 0 else None,
        top_k=args.k if args.k > 0 else None,
        num_beams=args.beams if args.beams > 0 else None,
        decoder_start_token_id = decoder_start_token_id, # tokenizer.encode("[answer]")[0]
        early_stopping=True,
        no_repeat_ngram_size=2,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=max(1, args.beams)
    )

    preds = [tokenizer.decode(
        out, skip_special_tokens=False, clean_up_tokenization_spaces=False) for out in outputs]

    output_ids = [input_ids[0][-1].item()] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
    output_ids = torch.tensor([output_ids]).to(device)
    decoder_input_ids = output_ids[:, :-1].contiguous()

    with torch.no_grad():
        lm_logits = model(input_ids, decoder_input_ids=decoder_input_ids, use_cache=False)[0]
    num_choices, out_length, vocab_size = lm_logits.shape
    lm_labels = output_ids[:, 1:].clone().contiguous()
    m = torch.nn.Softmax(dim=2)
    probs = m(lm_logits).view(-1, vocab_size)
    log_probs = [torch.log(probs[n][l]) for n, l in enumerate(lm_labels.view(-1))]

    return sum(log_probs) / len(log_probs), preds


def generate_regular(tokenizer, model, args, input, output, device):
    """
    Generate a sequence with models like GPT, GPT2, or XLNet
    """
    context_tokens = tokenizer.encode(input)
    max_length = args.max_length + len(context_tokens)
    input_ids = torch.tensor(context_tokens, device=device).unsqueeze(0)

    outputs = model.generate(
        input_ids=input_ids,
        do_sample=args.beams == 0,
        max_length=max_length,
        temperature=args.temperature,
        top_p=args.p if args.p > 0 else None,
        top_k=args.k if args.k > 0 else None,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=args.beams if args.beams > 0 else None,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=max(1, args.beams)
    )

    preds = [tokenizer.decode(out, skip_special_tokens=False)[len(input):].strip() for out in outputs]
    preds = [pred.split(".")[0] for pred in preds]

    process = lambda s: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
    example = process(input) + process(output)
    input_len = len(process(input))
    token_ids = torch.tensor([example]).to(device)
    with torch.no_grad():
        lm_logits = model(token_ids)[0]
    shift_logits = lm_logits[..., :-1, :].contiguous()
    batch_size, max_length, vocab_size = shift_logits.shape
    shift_labels = token_ids[..., 1:].clone().contiguous()
    m = torch.nn.Softmax(dim=2)
    probs = m(shift_logits).view(-1, vocab_size)
    log_probs = [torch.log(probs[n][l]) for n, l in enumerate(shift_labels.view(-1))]
    # only consider the label
    log_probs = log_probs[(input_len-1):-1]

    return sum(log_probs) / len(log_probs), preds


if __name__ == "__main__":
    main()
