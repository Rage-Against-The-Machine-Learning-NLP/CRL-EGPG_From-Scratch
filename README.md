## data processing
To process the dataset (required data pushed to `processed` dir as pickle files):
```sh
python -m src.process --dataset quora
```
available datasets are `quora` and `para`. 

**Note**: 
- the code runs in parallel with as many cores as possible, using the `multiprocessing` library 
- on a 16 core system with all cores being utilized, it took `quora` about *50 minutes* to process, and `para` about *9 hours* to process. 

`python -m src.dataset` for testing dataloader class
`python -m src.train` for training the model on the default config (`src/config.json`)

