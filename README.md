# DeepAcr: Predicting Anti-CRISPR with Deep Learning

Pytorch implementation and trained model for *AcrNET: Predicting Anti-CRISPR with Deep Learning*.

## Environment Setup
- Download the repository.
```python
git clone https://github.com/banma12956/AcrNET.git
cd AcrNET
```
- Install [anaconda](https://www.anaconda.com/) or [conda](https://docs.conda.io/projects/conda/en/latest/index.html).
- Create an environment for AcrNET and activate it.
```shell
conda env create -f environment.yml
conda activate AcrNET
```

## Run a simple demo

We provide a trained model and a simple dataset for demonstration. Run the following code and try:

```python
python test.py
```

## Train

We also provide the five-fold cross-validation training code we used in our experiment:

```python
python train_five_fold.py
```

The data we used in experiments can be downloaded at https://drive.google.com/file/d/1LK6y9g75ktlJEOy3CXZcPpZjQ4h4l-Ws/view?usp=sharing

## Data

You can also upload your own protein sequence data and use our trained model to make prediction.

DeepAcr needs structure information, evolutionary information and Transformer feature as input features, as shown in folder [data](https://github.com/banma12956/DeepAcr/tree/main/data). [RaptorX](http://raptorx.uchicago.edu/), [POSSUM](https://possum.erc.monash.edu/index.jsp), [ESM-1b](https://github.com/facebookresearch/esm) can provide corresponding calculation.
