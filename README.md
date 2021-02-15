# Fair Uniform Federated Learning

## dataset generatation:
* download the original adult dataset: 
`wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data`
`wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test`

* dataset split
`python dataset_generate.py`
the split dataset is in ./data/train and ./data/test

## synthetic experiment 
    `cd synthetic`
    `python experiment.py`
the generated results are in ./synthetic/figures/

## real-world dataset experiment

### unifrom fairness budget on both clients 
    
### specific farienss budget on both clients 

### Equal opportunity measurement with uniform fairness budget on both clients


