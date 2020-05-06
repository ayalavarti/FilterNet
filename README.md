# FilterNet
This project implements a deep learning architecture for unpaired image enhancement using reinforcement learning. 


## Configurations

Run `./init <--i> <--gcp>` with an optional `-i` argument to install the dependencies for the Python virtual environment and
an option `--gcp` flag to download the dataset from the GCP Storage bucket.

Run `pytest` from the root directory to run all unit tests.


## Training and Testing
To run the neural net model, use
```
python main.py [-h] [--checkpoint-dir] [--device] {train, test, evaluate, performance} ...
```
Use the `-h` or `--help` flags to view optional arguments for each command of `{train, test, evaluate, performance}`. For example `python main.py train -h`.

To train with default hyperparameters and settings, run
```
python main.py train
```

## Edit Local Images
With learned model weights in the specified `checkpoint-dir`, run 
```
python main.py evaluate --image-path IMAGE_PATH
```
where `IMAGE_PATH` is the path to any image stored locally to run the given image through the generator and display the resulting edits. 
