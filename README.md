# REV
Code for the paper [REV: Information-Theoretic Evaluation of Free-Text Rationales](https://arxiv.org/pdf/2210.04982.pdf)

### Preparation
Install the packages and toolkits in `requirements.txt`

### Construct Baseline Rationales
```
./run_question_converter.sh task dataset_path device
```
The input data should be in a `.jsonl` file with the format `{"question_text": "...?", "answer_text": "..."}`
The output will be saved in a `.jsonl.predictions` file with the format `{"question_text": "...?", "answer_text": "...", "question_statement_text": "..."}`. We use the `"question_statement_text"` as the baseline rationale.

### Training Base Models

**Training CNN/LSTM base models**

For IMDB, set `--max_seq_length 250`. Fine-tune hyperparameters (e.g. learning rate, the number of hidden units) on each dataset.
```
python train.py train --gpu_id 2 --model cnn/lstm --dataset sst2/imdb/ag/trec --task base --batch-size 64 --epochs 10 --learning-rate 0.01 --max_seq_length 50
```


### Acknowledgments
The code for constructing baseline rationales (for CQA task) was adapted from [jifan-chen/QA-Verification-Via-NLI](https://github.com/jifan-chen/QA-Verification-Via-NLI/tree/master/seq2seq_converter)


### Reference:
If you find this repository helpful, please cite our paper:
```bibtex
@article{chen2022rev,
  title={REV: Information-Theoretic Evaluation of Free-Text Rationales},
  author={Chen, Hanjie and Brahman, Faeze and Ren, Xiang and Ji, Yangfeng and Choi, Yejin and Swayamdipta, Swabha},
  journal={arXiv preprint arXiv:2210.04982},
  year={2022}
}
```
