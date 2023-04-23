# Coherence Boosting (ACL 2022)

Source code for the ACL 2022 paper "Coherence boosting: When your pretrained language model is not paying enough attention" ([arXiv](https://arxiv.org/abs/2110.08294), [ACL Anthology](https://aclanthology.org/2022.acl-long.565/))

<!-- [**DEMO (live during ACL conference)**](http://13.90.37.71:4444/cb.html)   -->

****

* <a href='#introduction'>1. Introduction</a>
* <a href='#citation'>2. Citation</a>
* <a href='#demo'>3. Demo: Contrastive Next Token Prediction</a>
* <a href='#lambada'>4. LAMBADA: Prediction of Words Requiring Long Context</a>
* <a href='#nlu'>5. Natural Language Understanding</a>
    * <a href='#other_datasets'>5.1. Apply Coherence Boosting to Your Own Multi-choice Datasets</a>
* <a href='#nlg'>6. Natural Language Generation</a>
* <a href='#contact'>7. Contact</a>

****


<span id='introduction'/>

#### 1. Introduction
Long-range semantic coherence remains a challenge in automatic language generation and understanding. We demonstrate that large language models have insufficiently learned the effect of distant words on next-token prediction. We present **Coherence Boosting**, an inference procedure that increases a LM’s focus on a long context. We show the benefits of coherence boosting with pretrained models by distributional analyses of generated ordinary text and dialog responses. It is also found that coherence boosting with state-of-the-art models for various zero-shot NLP tasks yields performance gains with no additional training.
****

<span id='citation'/>

#### 2. Citation
If you find the paper and code useful, please kindly star this repo and cite the paper. Thanks so much!

```bibtex
@inproceedings{malkin-etal-2022-coherence,
    title = "Coherence boosting: When your pretrained language model is not paying enough attention",
    author = "Malkin, Nikolay and Wang, Zhen and Jojic, Nebojsa",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.565",
    doi = "10.18653/v1/2022.acl-long.565",
    pages = "8214--8236"
}
```

****

<span id='demo'/>

#### 3. Demo: Contrastive Next Token Prediction
We present a demo program to demonstrate the lack of coherence on existing pre-trained LMs, i.e., failing to corectly predict the next token given a context, which clearly requires the understanding of distant words. Such errors can be resolved by our proposed **Coherence Boosting**, which contrastivly predicts the next token by log-linear mixing two distributions from the full context and a partial context. 

```python
>>> from cb_demo import contrasting
>>> contrasting(model_name='gpt2', context=' Ballad metre is "less regular and more conversational" than common metre', --partial_length=8, --alpha=0.5)

[out]
Top tokens based on full context:
 Ballad metre is "less regular and more conversational" than common

Rank    Tokens     Logprobs    Probs
------  ---------  ----------  ---------
1       Ġsense     -2.405      9.03%
2       Ġin        -3.900      2.02%
3       .          -3.978      1.87%
4       ,          -4.097      1.66%
5       Ġpractice  -4.287      1.37%
...     ...        ...         ...
13      Ġmetre**   -5.098      0.610609%
** Target Token


Top tokens based on partial context:
 regular and more conversational" than common

Rank    Tokens         Logprobs    Probs
------  -------------  ----------  ---------
1       Ġsense         -2.547      7.83%
2       ĠEnglish       -3.352      3.50%
3       .              -3.427      3.25%
4       Ġconversation  -3.445      3.19%
5       ,              -3.634      2.64%
...     ...            ...         ...
14103   Ġmetre**       -13.450     0.000144%
** Target Token


Contrastive next token prediction:

Rank    Tokens    Logprobs    Probs
------  --------  ----------  -------
1       Ġmetre**  -0.923      39.74%
2       Ġsense    -2.334      9.69%
3       Ġmeter    -2.785      6.17%
4       Ġin       -3.210      4.03%
5       Ġfoot     -3.220      3.99%
** Target Token
```

You can replicate the results for some examples in Figure 1 of the paper by the following code:
```python
python cb_demo.py --context=' Ballad metre is "less regular and more conversational" than common metre' --model_name='gpt2' --partial_length=8 --alpha=0.5

python cb_demo.py --context=' Isley Brewing Company: Going Mintal — a minty milk chocolate stout' --model_name='gpt2' --partial_length=8 --alpha=0.5

python cb_demo.py --context=' Other times anxiety is not as easy to see, but can still be just as debilitating' --model_name='gpt2' --partial_length=8 --alpha=0.5

```

****

<span id='lambada'/>

#### 4. LAMBADA: Prediction of Words Requiring Long Context

[LAMBADA](https://arxiv.org/abs/1606.06031) task is similar to examples shown above where the model is expected to predict the final word in passages of several sentences. This dataset is a standard benchmark to evaluate modern langauge models ([example](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/)). 

More importantly, this task explicitly requires reasoning over a broad context: humans can reliably guess the last word when given a whole passage, but not when given only the last sentence. Such a property makes this benchmark a perfect testbed to evaluate the effectiveness of our proposed **Coherence Boosting**. 

To run the LAMBADA experiments, simply run the following command:
```python
python main.py --tasks='lambada' --models='gpt2-small' --use_val=False --alpha_start=1 --alpha_end=1 --alpha_step=0.1 --slen_start=10 --slen_end=10
```
Some important arguments are listed as follows, please use `python main.py --help` to see a complete list.
* `--models`: The name of pre-trained language models, seperated by semicolon if you want to run multiple models at the same time, e.g., `'gpt2-small;gpt2-medium'`; if you want to use GPT-3 models, see <a href='#gpt3_notes'>notes about GPT-3</a>.
* `--use_val`: Whether to use a validation set to select two hyperparameters, `alpha` and `slen` representing the boosting coefficient and length for the partial context
* `--alpha_start`, `--alpha_end`, `--alpha_step`: Grid search parameters for the `alpha` hyperparameter
* `--slen_start`, `--slen_end`, `--slen_step`: Grid search parameters for the `slen` hyperparameter; note that both hyperparameter setups influence the inference speed for LAMBADA task

****

<span id='nlu'/>

#### 5. Natural Language Understanding

We evaluate the proposed **Coherence Boosting** on the following NLU tasks.
|Task|Close Task|Question Answering|Text Classification|NLI|Fact Knowledge Retrieval
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|Datasets|[StoryCloze](https://cs.rochester.edu/nlp/rocstories/)<br />[HellaSwag](https://rowanzellers.com/hellaswag/)<br />[COPA](https://super.gluebenchmark.com/tasks)|[CommonsenseQA](https://www.tau-nlp.org/commonsenseqa)<br />[OpenBookQA](https://allenai.org/data/open-book-qa)<br />[ARC Easy/Challenge](https://leaderboard.allenai.org/arc_easy)<br />[PIQA](https://yonatanbisk.com/piqa/)|[SST-2/5](https://gluebenchmark.com/tasks)<br />TREC<br />AGNews|[RTE](https://super.gluebenchmark.com/tasks)<br />[CB](https://super.gluebenchmark.com/tasks)<br />[BoolQ](https://super.gluebenchmark.com/tasks)|[LAMA](https://github.com/facebookresearch/LAMA)|

Most of datasets can be loaded by the Hugginface's [datasets](https://huggingface.co/datasets); only a few of them require manually downloading with instructions prompted when you run the code.

To run NLU experiments, simply run the following command:
```python
python main.py --tasks='storycloze;csqa;openbookqa' --models='gpt2-small;gpt2-medium;gpt2-large' --alpha_start=2 --alpha_end=-3 --alpha_step=0.01
```
Some important arguments are listed as follows, please use `python main.py --help` to see a complete list.
* `--models`: The name of pre-trained language models, seperated by semicolon if you want to run multiple models at the same time, e.g., `'gpt2-small;gpt2-medium'`
* `--use_val`: Whether to use a validation set to select two hyperparameters, `alpha` and `slen` representing the boosting coefficient and length for the partial context
* `--alpha_start`, `--alpha_end`, `--alpha_step`: Grid search parameters for the `alpha` hyperparameter; note that the code caches intermediate results and you are free to change these parameters after running the inference for one time

<span id='gpt3_notes'/>

##### Notes about GPT-3
1. If you want to run GPT-3 models, please put the API key to a file named `api_key.txt`
2. The GPT-3 results in our paper is based on the original base GPT-3 series that may have slightly different performance and parameter requirements compared with the recent [GPT-3 series](https://beta.openai.com/docs/engines/gpt-3).


<span id='other_datasets'/>

##### 5.1. Apply Coherence Boosting to Your Own Multi-choice Datasets

In addition to the previous tasks, our codebase is flexible enough to incorporate any new multi-choice dataset with minial efforts (inspired by the open-source project, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)). There roughly three steps:
1. Register the new dataset in `__init__.py` in the `tasks` folder.
2. Create a new class inheriting the `MultipleChoiceTask` class with data preprocessing functions (e.g., `load_data`, `standardize`)
3. The most important function is the `get_contrast_ctx`, which is where you define your own premise-free prompt for coherence boosting

See other task classes as examples and please free feel to let us know if you encounter any problems when adopting our code. 

****

<span id='nlg'/>

#### 6. Natural Language Generation

We provide a generation model wrapper compatible with the HuggingFace `transformers` library in `generation/generation.py`. You can create coherence-boosted variants of any autoregressive LM using the class in the [example script](https://github.com/zhenwang9102/coherence-boosting/tree/main/generation/generation.py):
```python
>>> boosted_model = generation.BoostedModel(base_model, k=8, alpha_long=1.5, alpha_short=-0.5)
```
The `boosted_model` can then be flexibly used with the `generate` function, 

```python
>>> ins = T.LongTensor([tokenizer.encode('Once upon a midnight dreary,')])
>>> outputs = boosted_model.generate(input_ids=ins, do_sample=True, max_length=100, top_p=0.95)
>>> tokenizer.decode(outputs[0])
"Once upon a midnight dreary, while I pondered over these things, I suddenly became aware of a strange and terrible noise. I turned round, and saw that the old man was standing near me. He was wearing a black suit, with a black tie, and a black hat. He had a long, thin, black beard, and his eyes were black. His hair was of a dark brown colour, and was very long. His face was rather large, and his lips were somewhat"
```  

The model wrapper is readily adapted to scenarios in which the short context is the currently generated text minus a prefix of a certain length (*e.g.*, the previous turn in a conversation) by dynamically setting `boosted_model.k` to the negative prefix length.

We present some [conditional generation outputs](https://github.com/zhenwang9102/coherence-boosting/tree/main/generation/outputs). The evaluation metrics shown in Table 1 can be evaluated using the code from [this repository](https://github.com/ari-holtzman/degen) for the first four columns or using the code [here](https://github.com/zhenwang9102/coherence-boosting/tree/main/generation/metrics) for the new long-range coherence metrics we introduced.

****

<span id='contact'/>

#### 7. Contact
If you have any questions, please feel free to contact Kolya (nikolay.malkin at mila.quebec) and Zhen (wang.9215 at osu.edu).

