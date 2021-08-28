# How we design the NER model?

<!-- Edit at https://hackmd.io/dT1MhCdhSrmECVDYTFCplg -->
<!-- Edit at https://app.grammarly.com/ddocs/1265777822 -->

we use [Adapter Transformers](https://docs.adapterhub.ml/quickstart.html#introduction) to train a scaleable NER model.

## What is Adapter?

![](https://i.imgur.com/P9Hz543.png)

<!-- ![](https://i.imgur.com/aVTwWrY.png) -->

Adapter was first proposed in 2019. It fixed the weight of Transformers' pre-train model, and insert a tiny down-stram model, so call Adapter, to interpret the output. When training, it will only do gradient descend on Adapter while keeping the Pre-Train model weight fix. 

The paper had shown that, by using Adapter, the performance is almost like the conventional methods that train the whole model. Yet, the training time for the former is far less than the latter because we only need to tune the little down-stram model with fewer parameters.

Furthermore, we can use [Parallel](https://docs.adapterhub.ml/adapter_composition.html?highlight=stack#parallel) calculation for multiple Adapters. That is, we can use many Adapters on prediction, and Adapters won't affect each other. For more detail, Please reference [the paper](https://arxiv.org/pdf/1902.00751.pdf).


![](https://i.imgur.com/TCSGnjl.png)

In the picture above, each Adapter represents a label in the NER model. And All of them are compute prediction parallelly using the same output from the BERT pre-train model, which weight keeps freezing.


![](https://i.imgur.com/SJLoBrZ.png)

When training, we will only do the training step (Gradient Descent) on the specific Adapter. 

Adapter Transformers' model supports hot reload. Thus, after training is finished, the model can immediately overwrite the old Adapter with the new one before making the prediction within a second. 

## FYI:
- [How Backend Train and Use NER Adapters?](./How_backend_train_and_use_NER_Adapters.md)
