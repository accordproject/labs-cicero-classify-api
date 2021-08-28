# How Backend Train and Use NER Adapters?

<!-- https://hackmd.io/WRN9UojJSPWEIa_REIca7Q -->
<!-- https://app.grammarly.com/ddocs/1265786531 -->

We use [Adapter-Transformers](https://adapterhub.ml/) to implement the NER model. 

## Independent training pipeline
When the backend service starts, it will execute [`NER_trainer_runner.py`](../trainer/NER_trainer_runner.py), a simple single-threaded python script. It is a task manager, will monitor the training jobs collection in DB. 

If there is any job that status is `waiting` or `training`, the [`NER_trainer_runner`](../trainer/NER_trainer_runner.py) will execute the [`NER_trainer.py`](../trainer/NER_trainer.py), which will train the NER Adapter.

[`NER_trainer.py`](../trainer/NER_trainer.py) will select the oldest job in the collection which status is `waiting` or `training`  to train and will change its status to `training`. 

After that, the trainer will filter out the data according to the job's `train_data_filter`, and start training. 

After the training step finished, the trainer will save the Adapters on the local folder specify by `NER_ADAPTERS_PATH` on [`config.py`](../core/config.py) and update the label collection's `adapter.current_adapter_filename` attribute. Then change the job's status to `done`. 

## Load adapters
All the Adapters will store on the local folder specified by `NER_ADAPTERS_PATH` on `config.py`, and its filename will be store into MongoDB according to its represent label name.

Then, the API backend will load them all and build a Parallel Output to get their predictions at one calculation.

Adapter Transformers' model supports hot reload. Thus, if there is any update, the ML model in API can immediately overwrite the old Adapter with the new one before making the prediction within a second. 

Moreover, the user can also select a specific Adapter Model version for the particular label by calling PUT `/api/v1/models/NER/labelText` or using the `specify_model_version` parameter when making prediction.

> FYI:  [How we design the NER model with Adapter?](./How_we_design_the_NER_model.md).

