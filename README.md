# Progressive-Concept-Bottleneck-Models
This is the official implementation of our [preprint](https://arxiv.org/pdf/2211.10630). 

## Environment
```
pip install -r requirements.txt
```

## Data Prepration
The model has three stages: **observer**, **conceiver**, and **predictor**. There are three Python files to train them separately. To train with your dataset, it would be great to start with writing the DataSet class. You may want to have a look at the FetalSeg class in `./datasets/Fetaltrim3.py`. Generally, the DataSet class should have these return values in a dictionary from the `​get_item()` method: 

- *'image'* or *'gray_image'*: the image of each instance

- *'mask'*: segmentation mask

- *'label'*: classification label

- *'concept'*: predefined property methods, should be a vector

Note that there should also be a method called `get​_labels()` in the class. This is used for resampling the unbalanced dataset. The return value *'label'* is the data classification label with the same length as the dataset. 

After you have added your DataSet class, you should add it to `./datasets/__init__.py`. Then, add them to `train_observer.py`, `train_conceiver.py`, and `train_predictor.py`.  This can be done by adding an *'elif'*.

 

The second step is to modify the training configurations. They are stored in YAML files in `./configs`.   

For the observer, an example is in `./configs/observer.yaml`:

**'DATA'** includes the dataset configs. It would help if you revised *'DataSet'* to your own dataset, which is by the modification you made to `train_observer.py`. *'Configs'* is the input parameters of your defined DataSet class. *'ClassNum'* is the number of Segmentation Classes, including the background. *'ImageChannel'* is the channel number of the input image, it can be 1 or 3. 

**'TRAINING'** includes the training configs. You can change the learning rate, epochs, random seed, and scheduler here. *'TrainAugmentations'* is the data augmentation used for training the segmentation model. *'TrainSize'* and *'EvalSize'* are the input image shapes during training and validation/testing. If *'UseSGD'* is True, SGD is used. Otherwise, AdamW is used. *'MonitorMetric'* is the metric used to save the best model. It should be inside *METRICS: EvalClassMetrics: []*. 

**'MODEL'** means the model configs. You can change the *'Backbone'* into either segmentation network. When you are not using DTU-Net, please revise the *'Loss'* in *'TRAINING'* to *'dicefocal loss: 1'* and the *'LossConfigs'* to *'dicefocal loss: {}'*. 

**'METRICS'** includes the metrics used to evaluate the model.


## Model Training

To train the observer, use

```
python train_observer.py --config=configs/observer.yaml
```

For the conceiver, you may want to modify `configs/conceiver.yaml`. 

Here, *'SegChannel'* means the number of segmentation categories. *'ImageChannel'* is the input image channel number. The prediction of the Conceiver is called the property concept. It can include binary concepts (0 or 1, means categorical values), or quality concepts (any number that needs regression instead of classification). *'RegIndex'* is the indices of the quality concepts in the full property concept vector. *'CatIndex'* means the indices of binary concepts. The property concepts can be the property of the full image, or of object(s) in the image. We call the property of the full image 'global concepts', and the property of objects, 'local concepts'. You can specify these by setting their indices of them at *'GlobalIndex'* and *'LocalIndex'*. If you want multiple subnetworks for predicting different groups of global or local concepts separately, just write the groups in a list. For example, if I have two groups of local concepts, I would write *'LocalIndex: [[0,1,2], [3,4,5]]'*. The code will construct a network for predicting concepts [0, 1, 2], and another one for predicting concepts [3,4,5]. You should also specify the reliance of local property concepts on the segmentation results. This is stored in a dictionary at *'Relationship'*. For example, if concept '1' relies on segmentation categories 2 and 3, the key-value pair should be {1: [2, 3]}. 

You should also modify the conceiver in `models/conceivers.py` by inheriting *'Conceiver'*

You only need to override the `seg_based_rule()` method. In this method, you can write some rules like 'if the kidney is not in the segmentation, the properties associated with the kidney should all be 0'.   

To train the conceiver, call 

```
python train_conceiver.py --config=configs/conceiver.yaml' 
```
 

For the predictor, revise `configs/predictor.yaml`. 

*'ConceptNum'* is the number of property concepts you defined for the conceiver. *'CatIndex'* has the same meaning as before. *'ClassNum'* is the number of classification categories. *'ExpandDim'* in **'MODEL'** means the number of neurons in the hidden layer.

 

To train it, call 

```
 python train_predictor.py --config=configs/predictor.yaml
 ```
 

 ## Evaluation

To evaluate the model performance, you may need to write a script similar to `eval_pcbm_fetal.py`. This includes a combination of the three stages. 
