# Boosting Coherence of Language Models (ACL 2022)

Source code for ACL 2022 paper, "[Coherence boosting: When your pretrained language model is not paying enough attention](https://arxiv.org/abs/2110.08294)"

****

* <a href='#introduction'>1. Introduction</a>
* <a href='#citation'>2. Citation</a>
* <a href='#demo'>3. Demo: Contrastive Next Token Prediction</a>
* <a href='#lambada'>4. LAMBADA: Prediction of Words Requiring Long Context</a>
* <a href='#nlg'>5. Natural Language Generation</a>
* <a href='#nlu'>6. Natural Language Understanding</a>
* <a href='#contact'>7. Contact</a>

****


<span id='introduction'/>

#### 1. Introduction
Long-range semantic coherence remains a challenge in automatic language generation and understanding. We demonstrate that large language models have insufficiently learned the effect of distant words on next-token prediction. We present Coherence Boosting, an inference procedure that increases a LM’s focus on a long context. We show the benefits of coherence boosting with pretrained models by distributional analyses of generated ordinary text and dialog responses. It is also found that coherence boosting with state-of-the-art models for various zero-shot NLP tasks yields performance gains with no additional training.
****

<span id='citation'/>

#### 2. Citation
If you find the paper and code useful, please kindy star this repo and cite the paper. Thanks so much!

```bibtex
@inproceedings{malkin2021boosting,
  author    = {Malkin, Nikolay and Wang, Zhen and Jojic, Nebojsa},
  title     = {Coherence boosting: When your pretrained language model is not paying enough attention},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year      = {2022}
}
```

****

<span id='demo'/>

#### 3. Demo: Contrastive Next Token Prediction
We present a demo program to demonstrate the lack of coherence on existing pre-trained LMs, i.e., failing to corectly predict the next token given a context, which clearly requires the understanding of distant words. Such errors can be resolved by our proposed **Coherence Boosting** as contrastivly predicting the next token by log-linear mixing two distributions from a long context and a partial context. 

```python
>>> from cb_demo import contrasting
>>> contrasting(model_name='gpt2', context='Ballad metre is "less regular and more conversational" than common metre', --partial_length=8, --alpha=0.5)

[out]
Top tokens based on full context:
Ballad metre is "less regular and more conversational" than common

Rank    Tokens     Logprobs    Probs
------  ---------  ----------  ---------
1       Ġsense     -2.418      8.91%
2       Ġin        -3.981      1.87%
3       ,          -4.059      1.73%
4       .          -4.066      1.72%
5       Ġpractice  -4.544      1.06%
...     ...        ...         ...
14      Ġmetre**   -5.206      0.548512%
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
1       Ġmetre**  -1.084      33.83%
2       Ġsense    -2.354      9.50%
3       Ġmeter    -2.800      6.08%
4       Ġfoot     -3.106      4.48%
5       Ġmeters   -3.209      4.04%
** Target Token
```

You can replicate the results for some examples in Figure 1 of the paper by the following code:
```python
python cb_demo.py --context='Ballad metre is "less regular and more conversational" than common metre' --model_name='gpt2' --partial_length=8 --alpha=0.5

python cb_demo.py --context='Isley Brewing Company: Going Mintal — a minty milk chocolate stout' --model_name='gpt2' --partial_length=8 --alpha=0.5

python cb_demo.py --context='Other times anxiety is not as easy to see, but can still be just as debilitating' --model_name='gpt2' --partial_length=8 --alpha=0.5

```

****

<span id='lambada'/>

#### 4. LAMBADA: Prediction of Words Requiring Long Context


****

<span id='nlg'/>

#### 5. Natural Language Generation
Generic Text

Dialogue Response Generation

****

<span id='nlu'/>

#### 6. Natural Language Understanding

|Task|Close Task|Question Answering|Text Classification|NLI|Fact Knowledge Retrieval
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|Datasets|StoryCloze<br />HellaSwag<br />COPA|CommonsenseQA<br />OpenBookQA<br />ARC Easy/Challenge<br />PIQA|SST-2/5<br />TREC<br />AGNews|RTE<br />CB<br />BoolQ|LAMA|


****

<span id='contact'/>

#### 7. Contact
If you have any questions, please feel free to contact Kolya (nikolay.malkin at mila.quebec) and Zhen (wang.9215 at osu.edu).

