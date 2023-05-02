import pandas as pd
import json
from ast import literal_eval
import random
import os

from transformers import AutoModelWithLMHead, AutoTokenizer

def init_model(model_name: str, device, do_lower_case: bool = False):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :param do_lower_case: whether the model is lower cased or not
    :return: the model and tokenizer
    """

    if model_name == 'bart-large':
        model_name = 'facebook/'+model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case) #, use_fast=False)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    # import pdb; pdb.set_trace()
    model.to(device)
    model.eval()
    return tokenizer, model


def load_data_ecqa(args, in_file, data_type=None, test_type=None, out_type=None):
      current_path = os.path.dirname(os.path.abspath(__file__))
      file_name = in_file.split('/')[-1]
      if 'train' in file_name:
            split_type = 'train'
      elif 'dev' in file_name:
            split_type = 'dev'
      elif 'test' in file_name:
            split_type = 'test'
      if args.do_train or test_type == 'gold':
            examples = []
            if data_type == 'regular':
                  # read baseline rationales (b)
                  template_file = os.path.join(current_path, '../', 'templated_rationales', args.task, split_type+'.jsonl.predictions')
                  template_file = os.path.normpath(template_file)
                  samples = []
                  with open(template_file, 'r') as json_file:
                        json_list = list(json_file)
                        for json_str in json_list:
                              result = json.loads(json_str)
                              label = result['answer_text']
                              rat = result['question_statement_text']
                              samples.append((rat, label))

                  # read gold rationales (r)
                  df = pd.read_csv(in_file)
                  for i, row in df.iterrows():
                        pos_rat = row['taskA_pos'].replace('\n', " ")
                        answer = row['q_ans']
                        sample = samples[i]
                        assert answer == sample[1]
                        # concatenate [r, b]
                        examples.append((f"[rationale] {pos_rat} {sample[0]} [answer]", f"{answer} <eos>"))
            elif data_type == 'temp':
                  with open(in_file, 'r') as json_file:
                        json_list = list(json_file)
                        for json_str in json_list:
                              result = json.loads(json_str)
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              # [b]
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))
      elif test_type == 'gen':
            # read the constructed baseline rationales for task model predicted labels
            template_file = os.path.join(current_path, '../task_model_output', args.task+'_'+args.out_type+'-'+args.task_model, \
                  split_type+'_'+args.out_type+'_baselines.jsonl.predictions')
            template_file = os.path.normpath(template_file)
            samples = []
            with open(template_file, 'r') as json_file:
                  json_list = list(json_file)
                  for json_str in json_list:
                        result = json.loads(json_str)
                        label = result['answer_text']
                        rat = result['question_statement_text']
                        samples.append((rat, label))

            examples = []
            with open(in_file, 'r') as json_file:
                  json_list = list(json_file)
            if out_type == 'YR':
                  i = -1
                  for json_str in json_list:
                        i += 1
                        result = json.loads(json_str)
                        if data_type == 'regular':
                              pred_rat = result['predictions'][0]
                              pred_rat = pred_rat.split('[answer]')[1].split('[rationale]')
                              if len(pred_rat) < 2:
                                    pred = pred_rat[0].split('<eos>')[0][1:-1]
                                    rat = ''
                              else:
                                    pred = pred_rat[0][1:-1]
                                    rat = pred_rat[1].split('<eos>')[0][1:-1]
                              sample = samples[i]
                              assert pred == sample[1]
                              examples.append((f"[rationale] {rat} {sample[0]} [answer]", f"{pred} <eos>"))
                        elif data_type == 'temp':
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))
            elif out_type == 'R':
                  for i, json_str in enumerate(json_list):
                        result = json.loads(json_str)
                        if data_type == 'regular':
                              input_label = result['input'].split('[answer]')
                              label = input_label[1].split('[rationale]')[0][1:-1]
                              rat = result['predictions'][0].split('[rationale]')[1].split('<eos>')[0][1:-1]
                              sample = samples[i]
                              assert label == sample[1]
                              examples.append((f"[rationale] {rat} {sample[0]} [answer]", f"{label} <eos>"))
                        elif data_type == 'temp':
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))
            elif out_type == 'RY':
                  i = -1
                  for json_str in json_list:
                        i += 1
                        result = json.loads(json_str)
                        if data_type == 'regular':
                              pred_rat = result['predictions'][0]
                              pred_rat = pred_rat.split('[rationale]')[1].split('[answer]')
                              if len(pred_rat) < 2:
                                    rat = pred_rat[0].split('<eos>')[0][1:-1]
                                    pred = ''
                              else:
                                    rat = pred_rat[0][1:-1]
                                    pred = pred_rat[1].split('<eos>')[0][1:-1]
                              sample = samples[i]
                              assert pred == sample[1]
                              examples.append((f"[rationale] {rat} {sample[0]} [answer]", f"{pred} <eos>"))
                        elif data_type == 'temp':
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))
      if args.do_train:
            random.shuffle(examples)
      return examples


def load_data_cose(args, in_file, data_type=None, test_type=None, out_type=None):
      current_path = os.path.dirname(os.path.abspath(__file__))
      file_name = in_file.split('/')[-1]
      if 'train' in file_name:
            split_type = 'train'
      elif 'dev' in file_name:
            split_type = 'dev'
      elif 'test' in file_name:
            split_type = 'test'
      if args.do_train or test_type == 'gold':
            examples = []
            if data_type == 'regular':
                  # read baseline rationales (b)
                  template_file = os.path.join(current_path, '../', 'templated_rationales', args.task, split_type+'.jsonl.predictions')
                  template_file = os.path.normpath(template_file)
                  samples = []
                  with open(template_file, 'r') as json_file:
                        json_list = list(json_file)
                        for json_str in json_list:
                              result = json.loads(json_str)
                              label = result['answer_text']
                              rat = result['question_statement_text']
                              samples.append((rat, label))

                  # read gold rationales (r)
                  df = pd.read_csv(in_file)
                  for i, row in df.iterrows():
                        pos_rat = row['rationale']
                        answer = row['answer']
                        sample = samples[i]
                        assert answer == sample[1]
                        # concatenate [r, b]
                        examples.append((f"[rationale] {pos_rat} {sample[0]} [answer]", f"{answer} <eos>"))
            elif data_type == 'temp':
                  with open(in_file, 'r') as json_file:
                        json_list = list(json_file)
                        for json_str in json_list:
                              result = json.loads(json_str)
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              # [b]
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))

      elif test_type == 'gen':
            # read the constructed baseline rationales for task model predicted labels
            template_file = os.path.join(current_path, '../task_model_output', args.task+'_'+args.out_type+'-'+args.task_model, \
                  split_type+'_'+args.out_type+'_baselines.jsonl.predictions')
            template_file = os.path.normpath(template_file)
            samples = []
            with open(template_file, 'r') as json_file:
                  json_list = list(json_file)
                  for json_str in json_list:
                        result = json.loads(json_str)
                        label = result['answer_text']
                        rat = result['question_statement_text']
                        samples.append((rat, label))

            examples = []
            with open(in_file, 'r') as json_file:
                  json_list = list(json_file)
            if out_type == 'YR':
                  i = -1
                  for json_str in json_list:
                        i += 1
                        result = json.loads(json_str)
                        if data_type == 'regular':
                              pred_rat = result['predictions'][0]
                              pred_rat = pred_rat.split('[answer]')[1].split('[rationale]')
                              if len(pred_rat) < 2:
                                    pred = pred_rat[0].split('<eos>')[0][1:-1]
                                    rat = ''
                              else:
                                    pred = pred_rat[0][1:-1]
                                    rat = pred_rat[1].split('<eos>')[0][1:-1]
                              sample = samples[i]
                              assert pred == sample[1]
                              examples.append((f"[rationale] {rat} {sample[0]} [answer]", f"{pred} <eos>"))
                        elif data_type == 'temp':
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))
            elif out_type == 'R':
                  for i, json_str in enumerate(json_list):
                        result = json.loads(json_str)
                        if data_type == 'regular':
                              input_label = result['input'].split('[answer]')
                              label = input_label[1].split('[rationale]')[0][1:-1]
                              rat = result['predictions'][0].split('[rationale]')[1].split('<eos>')[0][1:-1]
                              sample = samples[i]
                              assert label == sample[1]
                              examples.append((f"[rationale] {rat} {sample[0]} [answer]", f"{label} <eos>"))
                        elif data_type == 'temp':
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))
            elif out_type == 'RY':
                  i = -1
                  for json_str in json_list:
                        i += 1
                        result = json.loads(json_str)
                        if data_type == 'regular':
                              pred_rat = result['predictions'][0]
                              pred_rat = pred_rat.split('[rationale]')[1].split('[answer]')
                              if len(pred_rat) < 2:
                                    rat = pred_rat[0].split('<eos>')[0][1:-1]
                                    pred = ''
                              else:
                                    rat = pred_rat[0][1:-1]
                                    pred = pred_rat[1].split('<eos>')[0][1:-1]
                              sample = samples[i]
                              assert pred == sample[1]
                              examples.append((f"[rationale] {rat} {sample[0]} [answer]", f"{pred} <eos>"))
                        elif data_type == 'temp':
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))
      if args.do_train:
            random.shuffle(examples)
      return examples


def load_data_esnli(args, in_file, data_type=None, test_type=None, out_type=None):
      current_path = os.path.dirname(os.path.abspath(__file__))
      file_name = in_file.split('/')[-1]
      if 'train' in file_name:
            split_type = 'train'
      elif 'dev' in file_name:
            split_type = 'dev'
      elif 'test' in file_name:
            split_type = 'test'
      if args.do_train or test_type == 'gold':
            examples = []
            if data_type == 'regular':
                  # read baseline rationales (b)
                  template_file = os.path.join(current_path, '../', 'templated_rationales', args.task, split_type+'.jsonl.predictions')
                  template_file = os.path.normpath(template_file)
                  samples = []
                  with open(template_file, 'r') as json_file:
                        json_list = list(json_file)
                        for json_str in json_list:
                              result = json.loads(json_str)
                              label = result['answer_text']
                              rat = result['question_statement_text']
                              samples.append((rat, label))

                  # read gold rationales (r)
                  df = pd.read_csv(in_file)
                  for i, row in df.iterrows():
                        pos_rat = row['rationale']
                        answer = row['label']
                        sample = samples[i]
                        assert answer == sample[1]
                        # concatenate [r, b]
                        examples.append((f"[rationale] {pos_rat} {sample[0]} [answer]", f"{answer} <eos>"))
            elif data_type == 'temp':
                  with open(in_file, 'r') as json_file:
                        json_list = list(json_file)
                        for json_str in json_list:
                              result = json.loads(json_str)
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              # [b]
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))

      elif test_type == 'gen':
            # read the constructed baseline rationales for task model predicted labels
            template_file = os.path.join(current_path, '../task_model_output', args.task+'_'+args.out_type+'-'+args.task_model, \
                  split_type+'_'+args.out_type+'_baselines.jsonl.predictions')
            template_file = os.path.normpath(template_file)
            samples = []
            with open(template_file, 'r') as json_file:
                  json_list = list(json_file)
                  for json_str in json_list:
                        result = json.loads(json_str)
                        label = result['answer_text']
                        rat = result['question_statement_text']
                        samples.append((rat, label))

            examples = []
            with open(in_file, 'r') as json_file:
                  json_list = list(json_file)
            if out_type == 'YR':
                  i = -1
                  for json_str in json_list:
                        i += 1
                        result = json.loads(json_str)
                        if data_type == 'regular':
                              pred_rat = result['predictions'][0]
                              pred_rat = pred_rat.split('[answer]')[1].split('[rationale]')
                              if len(pred_rat) < 2:
                                    pred = pred_rat[0].split('<eos>')[0][1:-1]
                                    rat = ''
                              else:
                                    pred = pred_rat[0][1:-1]
                                    rat = pred_rat[1].split('<eos>')[0][1:-1]
                              sample = samples[i]
                              assert pred == sample[1]
                              examples.append((f"[rationale] {rat} {sample[0]} [answer]", f"{pred} <eos>"))
                        elif data_type == 'temp':
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))
            elif out_type == 'R':
                  for i, json_str in enumerate(json_list):
                        result = json.loads(json_str)
                        if data_type == 'regular':
                              input_label = result['input'].split('[answer]')
                              label = input_label[1].split('[rationale]')[0][1:-1]
                              rat = result['predictions'][0].split('[rationale]')[1].split('<eos>')[0][1:-1]
                              sample = samples[i]
                              assert label == sample[1]
                              examples.append((f"[rationale] {rat} {sample[0]} [answer]", f"{label} <eos>"))
                        elif data_type == 'temp':
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))
            elif out_type == 'RY':
                  i = -1
                  for json_str in json_list:
                        i += 1
                        result = json.loads(json_str)
                        if data_type == 'regular':
                              pred_rat = result['predictions'][0]
                              pred_rat = pred_rat.split('[rationale]')[1].split('[answer]')
                              if len(pred_rat) < 2:
                                    rat = pred_rat[0].split('<eos>')[0][1:-1]
                                    pred = ''
                              else:
                                    rat = pred_rat[0][1:-1]
                                    pred = pred_rat[1].split('<eos>')[0][1:-1]
                              sample = samples[i]
                              assert pred == sample[1]
                              examples.append((f"[rationale] {rat} {sample[0]} [answer]", f"{pred} <eos>"))
                        elif data_type == 'temp':
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))
      if args.do_train:
            random.shuffle(examples)
      return examples


def load_data_quartz(args, in_file, data_type=None, test_type=None, out_type=None):
      current_path = os.path.dirname(os.path.abspath(__file__))
      file_name = in_file.split('/')[-1]
      if 'train' in file_name:
            split_type = 'train'
      elif 'dev' in file_name:
            split_type = 'dev'
      elif 'test' in file_name:
            split_type = 'test'
      if args.do_train or test_type == 'gold':
            examples = []
            if data_type == 'regular':
                  # read baseline rationales (b)
                  template_file = os.path.join(current_path, '../', 'templated_rationales', args.task, split_type+'.jsonl.predictions')
                  template_file = os.path.normpath(template_file)
                  samples = []
                  with open(template_file, 'r') as json_file:
                        json_list = list(json_file)
                        for json_str in json_list:
                              result = json.loads(json_str)
                              label = result['answer_text']
                              rat = result['question_statement_text']
                              samples.append((rat, label))

                  # read gold rationales (r)
                  df = pd.read_csv(in_file)
                  for i, row in df.iterrows():
                        pos_rat = row['rationale']
                        answer = row['answer']
                        sample = samples[i]
                        assert answer == sample[1]
                        # concatenate [r, b]
                        examples.append((f"[rationale] {pos_rat} {sample[0]} [answer]", f"{answer} <eos>"))
            elif data_type == 'temp':
                  with open(in_file, 'r') as json_file:
                        json_list = list(json_file)
                        for json_str in json_list:
                              result = json.loads(json_str)
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              # [b]
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))

      elif test_type == 'gen':
            # read the constructed baseline rationales for task model predicted labels
            template_file = os.path.join(current_path, '../task_model_output', args.task+'_'+args.out_type+'-'+args.task_model, \
                  split_type+'_'+args.out_type+'_baselines.jsonl.predictions')
            template_file = os.path.normpath(template_file)
            samples = []
            with open(template_file, 'r') as json_file:
                  json_list = list(json_file)
                  for json_str in json_list:
                        result = json.loads(json_str)
                        label = result['answer_text']
                        rat = result['question_statement_text']
                        samples.append((rat, label))

            examples = []
            with open(in_file, 'r') as json_file:
                  json_list = list(json_file)
            if out_type == 'YR':
                  i = -1
                  for json_str in json_list:
                        i += 1
                        result = json.loads(json_str)
                        if data_type == 'regular':
                              pred_rat = result['predictions'][0]
                              pred_rat = pred_rat.split('[answer]')[1].split('[rationale]')
                              if len(pred_rat) < 2:
                                    pred = pred_rat[0].split('<eos>')[0][1:-1]
                                    rat = ''
                              else:
                                    pred = pred_rat[0][1:-1]
                                    rat = pred_rat[1].split('<eos>')[0][1:-1]
                              sample = samples[i]
                              assert pred == sample[1]
                              examples.append((f"[rationale] {rat} {sample[0]} [answer]", f"{pred} <eos>"))
                        elif data_type == 'temp':
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))
            elif out_type == 'R':
                  for i, json_str in enumerate(json_list):
                        result = json.loads(json_str)
                        if data_type == 'regular':
                              input_label = result['input'].split('[answer]')
                              label = input_label[1].split('[rationale]')[0][1:-1]
                              rat = result['predictions'][0].split('[rationale]')[1].split('<eos>')[0][1:-1]
                              sample = samples[i]
                              assert label == sample[1]
                              examples.append((f"[rationale] {rat} {sample[0]} [answer]", f"{label} <eos>"))
                        elif data_type == 'temp':
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))
            elif out_type == 'RY':
                  i = -1
                  for json_str in json_list:
                        i += 1
                        result = json.loads(json_str)
                        if data_type == 'regular':
                              pred_rat = result['predictions'][0]
                              pred_rat = pred_rat.split('[rationale]')[1].split('[answer]')
                              if len(pred_rat) < 2:
                                    rat = pred_rat[0].split('<eos>')[0][1:-1]
                                    pred = ''
                              else:
                                    rat = pred_rat[0][1:-1]
                                    pred = pred_rat[1].split('<eos>')[0][1:-1]
                              sample = samples[i]
                              assert pred == sample[1]
                              examples.append((f"[rationale] {rat} {sample[0]} [answer]", f"{pred} <eos>"))
                        elif data_type == 'temp':
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              examples.append((f"[rationale] {rat} [answer]", f"{answer} <eos>"))
      if args.do_train:
            random.shuffle(examples)
      return examples
