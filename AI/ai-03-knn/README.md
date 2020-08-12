# AI Assignment #03: k Nearest Neighbours

## A. Requirements
1. Python 3

## B. Running the Program
### 1. Cross Validation
To shuffle the training data `"DataTrain_Tugas3_AI.csv"`, use:
```python
> python -c "import knn; knn.shuffle_train_data()"
```
To do the cross validation, use:
```python
> python -c "import knn; knn.cross_validate([start_k], [end_k], [ratio], [shuffled])"
```
* [start_k] minimum k value.
* [end_k] maximum k value.
* [ratio] the ratio of the training data split. (default to `0.25`)
* [shuffled] whether to use shuffled data or not. If set to `True`, `"DataTrain_Tugas3_AI_Shuffled.csv"` will be used, otherwise the original training data will be used. (default to `True`)

### 2. Classify Test Data
To do the actual classification of the test data `"DataTest_Tugas3_AI.csv"`, use:
```python
> python knn.py
```
Then, simply enter the desired `k` value when asked.

The result will be written to `"TebakanTugas3.csv"` and `"TebakanTugas3_Detailed.csv"`.