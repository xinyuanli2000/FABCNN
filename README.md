# Feedback Attention-Based Deep CNN (FABCNN)
Deep CNN classifier models using Feedback Attention

## split_data.py

This script is to split the dataset.

## train.py

This script is to train the FABCNN, need to change training parameters, like epoch, iteration times, by yourself.

## test.py

This script is to test the classification accuracy and confusion matrix.

## ExecuteFeedbackAttentionCNN_mul.py

This script is improved by https://github.com/scajb/feedback-attention-cnn

This script loads a feedback attention model from file, then executes it against a specified input image/image directory. 

If bounding box XML annotations are available, these are also loaded.

The feedback attention model outputs a predicted class, which is reported to a log file. 

The model also returns a collection of its feedback activations, at different levels in the model. These are used to 
generate plots to visualise the model's attention regions, against the original input image.

The ExecuteFeedbackCNN functoin requires the following inputs. If running in PyCharm, these can be configured in the ExecuteFeedbackAttentionCNN run configuration. 

1. Path to pre-trained feedback attention model, as .PT file. Should be of one of the the PyTorch CNN implementations in the classes/classifier subdirectory of this project.
2. Path to input image/directory, e.g. from ImageNet-100 test set, as JPEG or PNG
3. Path to local log file output
4. Path to output directory, where generated images will be saved
5. Path to directory of XML files containing bounding box annotations

## change_color.py / change_position.py

The two script are used to creat new test dataset.

### Python environment

```
conda create --name feedback-attention-env python=3.7
source activate feedback-attention-env

conda install setuptools=45.2.0
pip install libarchive openslide-python
conda install pandas

pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install sklearn tqdm matplotlib scikit-image shapely descartes efficientnet_pytorch python-interface opencv-python
pip install -U jsonpickle
```

