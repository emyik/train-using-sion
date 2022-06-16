# Training using SION

## Deployment

Install Python 3.9
Install Visual Studio Code

~~~
pip3 install -r requirements.txt
~~~

## Play

Using Visual Studio to open train_cifar.ipynb, have fun.

## Run experiment

Test training:

~~~
make test
~~~

GPU Training. If GPU is not available, CPU will be used:

~~~
make train-cifar
~~~

### Output fields

Output will be in Comma Separated Values (CSV) format, fields desciption:

| Record Type                 | type | epoch     | start     | loading            | duration    | value1           | value2                 | Note                                          |
| --------------------------- | ---- | --------- | --------- | ------------------ | ----------- | ---------------- | ---------------------- | --------------------------------------------- |
| Downloading training data   | 10   | 0         | timestamp | data loading time  | total time  | # of data loaded | -                      | Download training data from S3                |
| Downloading validation data | 11   | 0         | timestamp | data loading time  | total time  | # of data loaded | -                      | Download training data from S3                |
| Training time per epoch     | 0    | epoch No. | timestamp | data loading time  | total time  | accuracy         | top5* accuracy         | *Correct if the label is in top5 predictions. |
| Validation time per epoch   | 1    | epoch No. | timestamp | data loading time  | total time  | accuracy         | top5* accuracy         | *Correct if the label is in top5 predictions. |
| Training time per batch     | 2    | epoch No. | timestamp | data loading time* | total time* | batch No.        | total batches in epoch | *Since epoch start                            |

Note:

1. The unit of duration is second. 1.023456 = 1 second 23 milliseconds 456 microseconds