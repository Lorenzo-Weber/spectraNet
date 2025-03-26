# Spectra NET for Classifying Crops

This project is a simple experiment based on the paper proposed by Zeit AI, which focuses on predicting soybeans contents with a CNN regressor. 

## My Approach

My approach was basically the same as in the paper, with the following differences:
- Number of neurons in the dense layers
- Weights initialization
- Learning rate
- I am not sure how their data augmentation process worked so i applied a simple gaussian noise
  into the data, repeated the process 10 times per predictor, increasing the dataset size 
  from 1200 instances all the way to 4.400. 

## Results

The network performed really well, achieving, on average:
- **82% precision** 
- **84% recall** 

## Paper Reference

The paper can be found at:  
[https://www.scitepress.org/publishedPapers/2024/126976/pdf/index.html](https://www.scitepress.org/publishedPapers/2024/126976/pdf/index.html)

