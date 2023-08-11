# REV
Code for the paper [REV: Information-Theoretic Evaluation of Free-Text Rationales](https://arxiv.org/pdf/2210.04982.pdf)

### Preparation
Create a conda environment:
```
conda env create -f rev_environment.yml
```
Activate the environment.

### Construct Baseline Rationales for CQA
In case of incompatibility, please use another environment with packages in [requirements](https://github.com/jifan-chen/QA-Verification-Via-NLI/blob/master/requirements.txt) to run the code
```
./run_question_converter.sh task dataset_path device
```
- The input data should be in a `.jsonl` file with the format `{"question_text": "...?", "answer_text": "..."}`
- The output will be saved in a `.jsonl.predictions` file with the format `{"question_text": "...?", "answer_text": "...", "question_statement_text": "..."}`. We use the `question_statement_text` as the baseline rationale.

### Construct Baseline Rationales for NLI
We first use a template to convert (premise, hypothesis, label) tuple into a baseline rationale: `premise implies/contradicts/is not related to
hypothesis`
```
python ./esnli_baseline/template.py
```
Then we paraphrase these templated, vacuous NLI rationales using a [pre-trained model](https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base)
```
python ./esnli_baseline/paraphrase.py
```

### Train the Evaluation Models
- Training $g$
```
bash ./rev/train.sh device regular task epochs learning_rate
```

- Training $g'$
```
bash ./rev/train.sh device temp task epochs learning_rate
```

### Compute REV
```
bash ./rev/evaluate.sh device split test_type out_type model_name task
```

### Acknowledgments
The code for constructing baseline rationales (for CQA task) was adapted from [jifan-chen/QA-Verification-Via-NLI](https://github.com/jifan-chen/QA-Verification-Via-NLI/tree/master/seq2seq_converter)


### Reference:
If you find this repository helpful, please cite our paper:
```bibtex
@inproceedings{chen-etal-2023-rev,
    title = "{REV}: Information-Theoretic Evaluation of Free-Text Rationales",
    author = "Chen, Hanjie  and
      Brahman, Faeze  and
      Ren, Xiang  and
      Ji, Yangfeng  and
      Choi, Yejin  and
      Swayamdipta, Swabha",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.112",
    pages = "2007--2030",
    abstract = "Generating free-text rationales is a promising step towards explainable NLP, yet evaluating such rationales remains a challenge. Existing metrics have mostly focused on measuring the association between the rationale and a given label. We argue that an ideal metric should focus on the new information uniquely provided in the rationale that is otherwise not provided in the input or the label. We investigate this research problem from an information-theoretic perspective using conditional V-information (Hewitt et al., 2021). More concretely, we propose a metric called REV (Rationale Evaluation with conditional V-information), to quantify the amount of new, label-relevant information in a rationale beyond the information already available in the input or the label. Experiments across four benchmarks with reasoning tasks, including chain-of-thought, demonstrate the effectiveness of REV in evaluating rationale-label pairs, compared to existing metrics. We further demonstrate REV is consistent with human judgments on rationale evaluations and provides more sensitive measurements of new information in free-text rationales. When used alongside traditional performance metrics, REV provides deeper insights into models{'} reasoning and prediction processes.",
}
```
