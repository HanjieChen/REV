import argparse
import csv
import os
import sys
import json
import pandas as pd


csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='0:gpu, -1:cpu')
parser.add_argument('--gpu_id', default='3', type=str, help='gpu id')
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

if args.test_type == 'gold':
        input_file = os.path.join(current_path, '../', 'data', args.task, args.split+'.csv')
        input_file = os.path.normpath(input_file)
        output_file = os.path.join(current_path, '../', 'templated_rationales', args.task, args.split+'.jsonl.predictions')
        output_file = os.path.normpath(output_file)

        df = pd.read_csv(input_file)
        examples = []
        for _, row in df.iterrows():
                label = row['label']
                premise = row['sentence_1']
                hypothesis = row['sentence_2']
                rationale = row['rationale']
                if label == 'entailment':
                        question_statement_text = premise + ' implies ' + hypothesis
                elif label == 'contradiction':
                        question_statement_text = premise + ' contradicts ' + hypothesis
                elif label == 'neutral':
                        question_statement_text = premise + ' is not related to ' + hypothesis
                examples.append(
                        {
                                'question_text': premise + ' ' + hypothesis,
                                'answer_text': label,
                                'question_statement_text': question_statement_text
                        }
                )

        with open(output_file, "w") as f_out:
                for example in examples:
                        f_out.write(
                                json.dumps(example)
                                + "\n"
                        )

elif args.test_type == 'gen':
        input_file = os.path.join(current_path, '../', 'task_model_output', args.task+'_'+args.out_type+'-'+args.task_model, \
                                  args.split+'_'+args.out_type+'_predictions.jsonl')
        input_file = os.path.normpath(input_file)
        output_file = os.path.join(current_path, '../', 'task_model_output', args.task+'_'+args.out_type+'-'+args.task_model, \
                                   args.split+'_'+args.out_type+'_predictions_r.jsonl.predictions')
        output_file = os.path.normpath(output_file)

        examples = []
        with open(input_file, 'r') as json_file:
                json_list = list(json_file)
        if args.out_type == 'YR':
                for json_str in json_list:
                        result = json.loads(json_str)
                        input = result['input'].split('[answer]')[0][:-1]
                        p_h = input.split('[premise]')[1].split('[hypothesis]')
                        premise = p_h[0][1:-1]
                        hypothesis = p_h[1][1:]
                        label_rat = result['predictions'][0]
                        label_rat = label_rat.split('[answer]')[1].split('[rationale]')
                        if len(label_rat) < 2:
                                label = label_rat[0].split('<eos>')[0][1:-1]
                        else:                            
                                label = label_rat[0][1:-1]
                        if label == 'entailment':
                                question_statement_text = premise + ' implies ' + hypothesis
                        elif label == 'contradiction':
                                question_statement_text = premise + ' contradicts ' + hypothesis
                        elif label == 'neutral':
                                question_statement_text = premise + ' is not related to ' + hypothesis
                        examples.append(
                                {
                                        'question_text': premise + ' ' + hypothesis,
                                        'answer_text': label,
                                        'question_statement_text': question_statement_text
                                }
                        )
        elif args.out_type == 'R':
                for json_str in json_list:
                        result = json.loads(json_str)
                        input_label = result['input'].split('[rationale]')[0].split('[answer]')
                        label = input_label[1][1:-1]
                        input = input_label[0][:-1]
                        p_h = input.split('[premise]')[1].split('[hypothesis]')
                        premise = p_h[0][1:-1]
                        hypothesis = p_h[1][1:]
                        if label == 'entailment':
                                question_statement_text = premise + ' implies ' + hypothesis
                        elif label == 'contradiction':
                                question_statement_text = premise + ' contradicts ' + hypothesis
                        elif label == 'neutral':
                                question_statement_text = premise + ' is not related to ' + hypothesis
                        examples.append(
                                {
                                        'question_text': premise + ' ' + hypothesis,
                                        'answer_text': label,
                                        'question_statement_text': question_statement_text
                                }
                        )
        elif args.out_type == 'RY':
                for json_str in json_list:
                        result = json.loads(json_str)
                        input = result['input'].split('[rationale]')[0][:-1]
                        p_h = input.split('[premise]')[1].split('[hypothesis]')
                        premise = p_h[0][1:-1]
                        hypothesis = p_h[1][1:]
                        label = result['predictions'][0].split('[rationale]')[1].split('[answer]')[1].split('<eos>')[0][1:-1]
                        if label == 'entailment':
                                question_statement_text = premise + ' implies ' + hypothesis
                        elif label == 'contradiction':
                                question_statement_text = premise + ' contradicts ' + hypothesis
                        elif label == 'neutral':
                                question_statement_text = premise + ' is not related to ' + hypothesis
                        examples.append(
                                {
                                        'question_text': premise + ' ' + hypothesis,
                                        'answer_text': label,
                                        'question_statement_text': question_statement_text
                                }
                        )

        with open(output_file, "w") as f_out:
                for example in examples:
                        f_out.write(
                                json.dumps(example)
                                + "\n"
                        )