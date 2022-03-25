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
We present a simply demo program to demonstrate the lack of coherence on existing pre-trained LMs, i.e., they may fail to corectly predict the next token given a context that clearly requires the understanding of distant words. Such errors can be resolved by our proposed **Coherence Boosting** by contrastivly predict the next token, log-linear mixing two distributions from a long context and a partial context. 

```python
# replicate some examples in Figure 1 of the paper

# Ballad metre is “less regular and more conversational” than common ?

# Isley Brewing Company: Going Mintal – a minty milk chocolate ?

# Other times anxiety is not as easy to see, but can still be just as ?

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

