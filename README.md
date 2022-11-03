# Credit

The original authors are

- Kang Wei
- Jun Li
- Chuan Ma
- Ming Ding
- Sha Wei
- Fan Wu
- Guihai Chen
- Thilina Ranbaduge

Corresponding paper can be found here: https://arxiv.org/abs/2202.04309

Original github repository can be found here: https://github.com/AdamWei-boop/Vertical_FL

# Changes from me

There were some errors which probably stem from not having the exact same data.
The data was not included in the original zip, so I added it myself to the best of my ability.

Finally I was able to reproduce everything with only minor changes to the code.

# Multi-headed Vertical Federated Neural Network

Prerequisites
-----
    Python 3.8
    Torch 1.8.0 or Tensorflow 2.3.0
Models&Data
-----
    Learning models: neural networks
    Datasets: Adult
Main Parameters
-----
    parser.add_argument('--dname', default='ADULT', help='dataset name')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs') 
    parser.add_argument('--batch_type', type=str, default='mini-batch')  
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--data_type', default='original', help='define the data options: original or one-hot encoded')
    parser.add_argument('--model_type', default='vertical', help='define the learning methods: vrtical or centralized')    
    parser.add_argument('--organization_num', type=int, default='3', help='number of origanizations, if we use vertical FL')    
Get Started
-----
    python tf_vertical_FL_train.py --epochs 100 --model_type 'vertical' --organization_num 3
    python torch_vertical_FL_train.py --epochs 100 --model_type 'vertical' --organization_num 3
    
