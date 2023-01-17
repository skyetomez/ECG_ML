# **ECG Heartbeat Categorization**

This was something I did for fun and I am doing the write up as well as improving on this project in my spare time. Feel free to fork or give suggetions.

The data was collected pre-labelled from [kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

There is a data augmentation function available in the preprocessing folder

Question is: Can you identify myocardial infarction?

## Requirements

Python: 3.10+

To install required packages:

```setup
pip install -r requirements.txt
```

## Datasets

[ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

## Data Cleaning

- Data was pre-cleaned and loaded in csv format
- data augmented using custom augmentation functions

## Training

- done on a 2x NVIDIA Tesla V100 (16GB VRAM)
- 75 epochs, batch_size = num_of_samples,
- learning schedule from paper
- optimizer from paper

## Evaluation

## Results

## Citations

- Mohammad Kachuee, Shayan Fazeli, and Majid Sarrafzadeh. "ECG Heartbeat Classification: A Deep Transferable Representation." *[arXiv preprint arXiv:1805.00794 (2018)](https://arxiv.org/abs/1805.00794)*.
- Wen, Qingsong, et al. *Time Series Data Augmentation for Deep Learning: A Survey*. 2020, doi:10.48550/ARXIV.2002.12478.
- Dilmegani, Cem. “Top Data Augmentation Techniques: Ultimate Guide for 2023.” *AIMultiple*, 30 Apr. 2021, [Top Data Augmentation Techniques: Ultimate Guide for 2023](https://research.aimultiple.com/data-augmentation-techniques/).

## Acknowledgements

- Approaches to this problem has been inspired by reading other people’s approaches to solving questions like this from kaggle, github, and papers. I encourage others to use my code in the same way.
