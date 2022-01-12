# DeepAcr: Predicting Anti-CRISPR with Deep Learning

Pytorch implementation and trained model for *DeepAcr: Predicting Anti-CRISPR with Deep Learning*.

## Dependencies
- Python 3.6+.
- PyTorch 1.10+. 

## Usage

We provide a trained model and a simple dataset for demonstration. Run the following code and try:

```python
python test.py
```

## Data

You can also upload your own protein sequence data and use our trained model to make prediction.

DeepAcr needs structure information, evolutionary information and Transformer feature as input features, as shown in folder [data](https://github.com/banma12956/DeepAcr/tree/main/data). [RaptorX](http://raptorx.uchicago.edu/), [POSSUM](https://possum.erc.monash.edu/index.jsp), [ESM-1b](https://github.com/facebookresearch/esm) can provide corresponding calculation.
