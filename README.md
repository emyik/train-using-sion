# Optimizing ResNet-50 CNN Data Loading and Training Speeds on CIFAR-10 Dataset

## Project Decription
The goal of this project is to evaluate different factors affecting machine learning data loading and training speeds. We optimized the ResNet-50 CNN training on the CIFAR-10 dataset, with results depicted in `plotData.ipynb`. 

Different types of trainings tested:
* Local CPU loading data from built-in version provided by torchvision.
* Google Colab T4 GPU loading data from built-in version provided by torchversion.
* Google Colab T4 GPU and loading dataset from S3.
* IBM VM IBM COS.
* IBM VM IBM COS with workers.


## How To Run Project

Deployment: Install Python 3.9 and Visual Studio Code.

~~~
pip3 install -r requirements.txt
~~~

Use Visual Studio to run train_cifar.ipynb.

To run the experiment:

~~~
make test
~~~

To run GPU Training (if GPU is not available, CPU will be used):

~~~
make train-cifar
~~~
### Output fields

The output is in a Comma Separated Values (CSV) format, as depicted below:

| Record Type                 | type | epoch     | start     | loading            | duration    | value1           | value2                 | Note                                          |
| --------------------------- | ---- | --------- | --------- | ------------------ | ----------- | ---------------- | ---------------------- | --------------------------------------------- |
| Downloading training data   | 10   | 0         | timestamp | data loading time  | total time  | # of data loaded | -                      | Download training data from S3                |
| Downloading validation data | 11   | 0         | timestamp | data loading time  | total time  | # of data loaded | -                      | Download training data from S3                |
| Training time per epoch     | 0    | epoch No. | timestamp | data loading time  | total time  | accuracy         | top5* accuracy         | *Correct if the label is in top5 predictions. |
| Validation time per epoch   | 1    | epoch No. | timestamp | data loading time  | total time  | accuracy         | top5* accuracy         | *Correct if the label is in top5 predictions. |
| Training time per batch     | 2    | epoch No. | timestamp | data loading time* | total time* | batch No.        | total batches in epoch | *Since epoch start                            |

Note: The unit of duration is second. 1.023456 = 1 second 23 milliseconds 456 microseconds
