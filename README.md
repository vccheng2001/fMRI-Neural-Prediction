
## Predicting Word Stimulus from fMRI Neural Activations: Gaussian Naive Bayes Classifier


Studies have shown that thinking about different semantic categories of words (for example, tools, buildings,and animals) activates 
different spatial patterns of neural activation in the brain. 

Given an fMRI image of 21764 voxels, this Gaussian Naive Bayes classifier predicts the associated stimulus word/class given to a human subject 
based on their observed neural activity measured by functional magnetic resonance imaging (fMRI).

## References 
This was developed for course 10-601 Machine Learning at Carnegie Mellon University in Spring 2021, taught by Professor Tom Mitchell and Matt Gormley. 

The data and experiments originated from a research paper from Professor Tom Mitchell et al: "Predicting Human Brain Activity Associated with the Meanings of Nouns"
For more information visit http://www.cs.cmu.edu/~tom/science2008/index.html.


## Usage 
To run the Gaussian Naive Bayes classifier:
```python
python3 gnb.py train_data.csv test_data.csv train_out.labels test_out.labels
```

To visualize the 2D slices of the brain: 
```python
python3 visualize.py <path/to/dataset> <row_index_into_dataset>
```

![screenrecording](https://user-images.githubusercontent.com/60167936/115097155-5f5fb180-9edd-11eb-9955-27a34972a950.gif)
