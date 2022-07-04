# GANforArt
**An implementation of "CAN: Creative Adversarial Networks"**  

**The aim of this repo is to implement a CAN model that can handle high-resolution images.**


## Data
Download wikiart data. [wikiart](https://www.wikiart.org/) dataset 
[available here](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset). 
Using the dataset is subject to wikiart's [terms of use](https://www.wikiart.org/en/terms-of-use)
~~~
mkdir data
cd data
mv DOWNLOADED_DATA_PATH ./
unzip wikiart.zip
~~~

## Installation
Install packages required to execute the code.  
~~~
$ pip install -r requirements.txt
~~~

## Train
You can train the model with the command below.
~~~
$ python main.py
~~~

## Test
You can test the model with the command below.
The trained models are saved in the "checkpoints" directory.  
~~~
$ python main.py --test --load-from TRAINED_MODEL_PATH
~~~

### Experiments


## References
[Creative Adversarial Networks](https://github.com/mlberkeley/Creative-Adversarial-Networks)  
[AEGeAN](https://github.com/tymokvo/AEGeAN)
