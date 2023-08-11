# https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base
# need new version of transformers
import argparse
import csv
import os
import sys
import json
import tqdm
import re

import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='0:gpu, -1:cpu')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--task', type=str, default='ESNLI', help='task name (ESNLI)')
parser.add_argument('--rat_type', default="pos", type=str, help="Use free flow rationales ('ff') or positive rationales ('pos') for training")
parser.add_argument('--test_type', default="gen", type=str, help="gold: gold rationales, gen: generative rationales")
parser.add_argument("--out_type", default="YR", type=str, help="Output type: 'YR' label and rationale, 'Y' label only, 'R' rationale only, 'RY' rationale and label")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="train, val, test",
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
        help="temp: b, regular: [r, b]",
    )
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if args.gpu > -1:
    args.device = "cuda"
else:
    args.device = "cpu"

current_path = os.path.dirname(os.path.abspath(__file__))


tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
model.to(args.device)

def paraphrase(
    question,
    args,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=200
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(args.device)
    
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res


if args.test_type == 'gold':
        input_file = os.path.join(current_path, '../', 'templated_rationales', args.task, args.split+'.jsonl.predictions')
        input_file = os.path.normpath(input_file)
        output_file = os.path.join(current_path, '../', 'templated_rationales', args.task, args.split+'.jsonl.predictions_r')
        output_file = os.path.normpath(output_file)
elif args.test_type == 'gen':
        input_file = os.path.join(current_path, '../', 'task_model_output', args.task+'_'+args.out_type+'-'+args.task_model, \
                args.split+'_'+args.out_type+'_predictions_r.jsonl.predictions')
        input_file = os.path.normpath(input_file)
        output_file = os.path.join(current_path, '../', 'task_model_output', args.task+'_'+args.out_type+'-'+args.task_model, \
                args.split+'_'+args.out_type+'_predictions_r.jsonl.predictions_r')
        output_file = os.path.normpath(output_file)

output_list = []
with open(input_file, 'r') as json_file:
        json_list = list(json_file)
        for json_str in tqdm.tqdm(json_list):
                result = json.loads(json_str)
                rat = result["question_statement_text"]
                rat = re.sub(r'[^\w\s]', '', rat)
                rat += '.'
                rat_r = random.choice(paraphrase(rat, args))
                output_list.append(
                        {
                        'question_text': result['question_text'],
                        'answer_text': result['answer_text'],
                        'question_statement_text': rat_r
                        }         
                )


with open(output_file, "w") as f_out:
        for example in output_list:
                f_out.write(
                        json.dumps(example)
                        + "\n"
                )
