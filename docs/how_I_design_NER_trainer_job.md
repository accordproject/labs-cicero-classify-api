# How I design NER trainer job?
When the backend service starts, it will execute [`NER_trainer_runner.py`](../trainer/NER_trainer_runner.py), a simple single-threaded python script. 

The [`NER_trainer_runner`](../trainer/NER_trainer_runner.py) will monitor the training jobs collection in DB. 

If there is any job which is `waiting` or `training`, the [`NER_trainer_runner`](../trainer/NER_trainer_runner.py) will execute the [`NER_trainer.py`](../trainer/NER_trainer.py), which will train the NER Adapter.

[`NER_trainer.py`](../trainer/NER_trainer.py) will select the oldest job in the collection which status is `waiting` or `training`  to train, and will change it's status to `training`. 

After that, the trainer will filter out the data according to the job's `train_data_filter`, and start training. 

After the training step is finished, trainer will save the adapters on the local folder specify by `NER_ADAPTERS_PATH` on [`config.py`](../core/config.py) and update the label db's `adapter.current_adapter_filename` attribute.