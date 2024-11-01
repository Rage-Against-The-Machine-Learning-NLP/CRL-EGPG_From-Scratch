## data processing
available datasets are `quora` and `para`. 

To process the dataset (required data pushed to `processed` dir as pickle files):
```sh
python -m src.process --dataset quora
```
it executes in parallel by default, add the `sequential` flag if you only want one worker. 
This step is not very necessary considering we've alr provided the cleaned data. 

`python src/dataset.py` for testing dataloader class