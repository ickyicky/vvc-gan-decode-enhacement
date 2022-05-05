# VVC quallity enhancement using GAN networks

This project is main object of my master's thesis. Following chapters describe how to reproduce my results

## Prerequisites

Project requires wide range of different tools used for data preparation, VVC compression and ML

### Tools

Following binaries should be installed:

- python3 is required for some scripts and ML.
- ffmpeg is used for video convertion from mkv to yuv format.
- mediainfo is used for video feature extraction

### Python libraries

List if python required libraries is in `requirements.txt`. You can install them manually or simply run:

```sh
python3 -m pip install -r requirements.txt
```

### VVC reference codec

For ease of use script `bin/fetch_vvc.sh` was created, it clones reference software and compiles it. Please run the script from the main project directory.

Simply run:

```sh
./bin/fetch_vvc.sh
```

The script will clone reference codec to vvc folder and compile everything for you.

### Dataset fetching and preparation

Dataset should be downloaded into `data` folder. `bin/fetch_dataset.py` was created so that only selected videos are downloaded. Simply run:

```sh
python3 bin/fetch_dataset.py
```

Script will create data folder download all the required data into it.

After the data is downloaded, it has to be converted to yuv format for encoder. Additionally, videos metadatas have to be collected. Another script was written for that purpose:

```sh
./bin/prepare_data.sh
```

### Dataset encoding and decoding

Next step is to encode and decode data using VVC with multiple parameters. For that purpose `bin/create_tasks.py` was written, generating list of jobs that have to executed. Run:

```sh
python3 bin/create_tasks.py TASKS
```

in order to create `TASKS` file containing list of jobs to run. You can either run them manually one by one or use `bin/pool_work.py` which manages execution of tasks:

```sh
python3 bin/pool_work.py TASKS DONE UNDONE --cpu-count=32
```

Where TASKS is special file containing jobs to run (shell instruction, each in new line generated in previous step), DONE is file which will be created if the process is interrupted of finished, containing list of completed jobs, UNDONE will contain remaining (unfinished) jobs and cpu-count means number of concurrent jobs to run (if not given defaults to number of CPU cores in the system).
