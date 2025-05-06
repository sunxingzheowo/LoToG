

# LoToG

### Requirements
- ``python 2.30``
- ``PyTorch 3.12``
- ``CUDA 12.1``
- ``transformers``
- ``matplotlib``
- `` sklearn``
- ``tqdm``
- ``numpy``

## Datasets
We experiment our model on two few-shot relation extraction datasets,
 1. [FewRel 1.0](https://thunlp.github.io/1/fewrel1.html)
 2. [FewRel 2.0](https://thunlp.github.io/2/fewrel2_da.html)

Please download data from the official links and put it under the ``./data/``. 

## Train

To run our model, use command

```bash
python train.py
```

This will start the training and evaluating process of LoToG in a 10-way-1-shot setting. You can also use different args to start different process. Some of them are here:

* `train / val / test`: Specify the training / validation / test set.
* `trainN`: N in N-way K-shot. `trainN` is the specific N in training process.
* `N`: N in N-way K-shot.
* `K`: K in N-way K-shot.
* `Q`: Sample Q query instances for each relation.

There are also many args for training (like `batch_size` and `lr`) and you can find more details in our codes.

#### fewrel1.0:

```bash
python train.py \
    --val val_wiki --test val_wiki --ispubmed False\
    --N 10 --K 1 --Q 1 --train_iter 30000 --val_iter 1000 --val_step 1000\
    --model LoToG --lr 2e-5 --grad_iter 1 --pretrain_ckpt bert_model\
    --only_test False --test_online False
```

#### fewrel2.0:

```bash
python train.py \
    --val val_pubmed --test val_pubmed --ispubmed True\
    --N 10 --K 1 --Q 1 --train_iter 30000 --val_iter 1000 --val_step 1000\
    --model LoToG --lr 2e-5 --grad_iter 1 --pretrain_ckpt bert_model\
    --only_test False --test_online False
```



## Evaluation


**FewRel 1.0**
```bash
python train.py \
    --N 10 --K 1 --Q 1 --test_iter 10000 --val_iter 1000 --val_step 1000\
    --model LoToG --lr 2e-5 --grad_iter 1 --pretrain_ckpt bert_model\
    --only_test True --test_online False
    --load_ckpt "checkpoint-new/my.pth.tar"
```

**FewRel 2.0**

```bash
python train.py \
    --val val_pubmed --test val_pubmed --ispubmed True\
    --N 10 --K 1 --Q 1 --test_iter 10000 --val_iter 1000 --val_step 1000\
    --model LoToG --lr 2e-5 --grad_iter 1 --pretrain_ckpt bert_model\
    --only_test True --test_online False
    --load_ckpt "checkpoint-new/my.pth.tar"
```
**Cited**

@inproceedings{sun-chen-2025-enhancing,
    title = "Enhancing the Prototype Network with Local-to-Global Optimization for Few-Shot Relation Extraction",
    author = "Sun, Hui  and
      Chen, Rongxin",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.145/",
    pages = "2668--2677",
    ISBN = "979-8-89176-195-7",
    abstract = "Few-Shot Relation Extraction (FSRE) aims to achieve high classification performance by training relation classification models with a small amount of labeled data. Prototypical networks serve as a straightforward and efficient method for optimizing model performance by combining similarity evaluation and contrastive learning. However, directly integrating these methods can introduce unpredictable noise, such as information redundancy, which hinders classification performance and negatively affects embedding space learning. The technique presented in this paper applies Local-To-Global optimization to enhance prototypical networks in few-shot relation extraction. Specifically, this paper develops a local optimization strategy that indirectly optimizes the prototypes by optimizing the other information contained within the prototypes. It considers relation prototypes as global anchors and incorporates the techniques introduced in this paper, such as information alignment, local contrastive learning, and a local adaptive focal loss function, to address the issues of information redundancy. This approach enables the model to learn a unified and effective embedding space. We conduct extensive experiments on the FewRel 1.0 and FewRel 2.0 datasets to validate the effectiveness of the proposed model."
}






