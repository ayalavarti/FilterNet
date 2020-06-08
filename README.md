# FilterNet

## Run Specifications

Run
```
./init <--i> <--gcp>
```
with an optional `-i` argument to install the dependencies for the Python virtual environment and
an option `--gcp` flag to download the dataset from the GCP Storage bucket.


Run `pytest` from the root directory to run all unit tests.

To run the neural net model, use
```
python main.py [-h] [--checkpoint-dir] [--device] {train, test} ...
```
Use the `-h` or `--help` flags to specify optional arguments. For example `python main.py train -h`.

## Web Commands
```
export FLASK_APP=app.py
flask run
```

## Postgres Commands
```
postgres -U <username>
\c filternet
\d models
```