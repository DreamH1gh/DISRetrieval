# RST Parser

## Folder structure
```
rst_parser/
├── config/
│   └── new.train (train config file)
├── experiment (train data folder)
├── saved_model (model save path)
├── driver/
│   ├── RSTparser.py (rst parser class, for inference)
│   ├── Test.py
│   └── TrainTest.py (train script)
├── xlnet (xlnet model path)
├── run.sh (training startup script)
└── ...
```

## Train
- Update the `config/new.train` file. Correctly set the `xlnet_dir` to the XLNet model path and the `save_dir` to the save path. `percentage` means the percentage of training data to be used.

- Modify the `train_percent` configuration in `run.sh` to specify the percentage of training data to be used.

- 
    ```
    sh run.sh
    ``` 
    to start training.

## Usage
- Set the default value of the `config_file` parameter in the `RSTParser` class (located in `driver/RSTParser.py`) to the path of the `config.cfg` file for the saved model you want to use.