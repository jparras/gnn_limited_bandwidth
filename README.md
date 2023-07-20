# A Graph Network Model for Distributed Learning with Limited Bandwidth Links and Privacy Constraints

## Introduction

Code used to obtain the results in the paper Parras, J., & Zazo, S. (2020, May). A Graph Network Model for Distributed Learning with Limited Bandwidth Links and Privacy Constraints. In 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 3907-3911). IEEE. [DOI](https://doi.org/10.1109/ICASSP40776.2020.9053067).

## Launch

This project has been tested on Python 3.6 on Ubuntu 18. To run this project, create a `virtualenv` (recomended) and then install the requirements as:

```
$ pip install -r requirements.txt
```

To show the results obtained in the paper, simply run the main file as:
```
$ python main.py
```

In case that you want to train and/or test, set the train and/or test flag to `True` in the `main_dgm.py` file and then run the same order as before. Note that the results file will be overwritten. 
