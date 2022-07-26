'''
# Time   : 2021/10/25 14:40
# Author : adamwei
# Vertical FL
'''

import numpy as np
import tensorflow as tf
import time
import argparse
import matplotlib.pyplot as plt

from deepFM_model import tf_organization_graph, tf_top_graph, DeepFM
from utils import create_criteo_dataset, batch_split
from tensorflow.keras import optimizers, losses
from sklearn.metrics import accuracy_score

def main(args):    
    
    data_type = args.data_type           # Define the data options: 'original', 'encoded'
    model_type = args.model_type         # Define the learning methods: 'vertical', 'centralized'
    epochs = args.epochs                 # number of training epochs
    #nrows = None                        # subselection of rows to speed up the program
    
    # Ming added the following variables for configurable vertical FL
    organization_num = args.organization_num    # number of participants in vertical FL

    k = 10
    w_reg = 1e-4
    v_reg = 1e-4

    # file_path = './datasets/criteo.txt'
    file_path = './datasets/criteo.csv'
        
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file_path, data_size=50000, test_size=0.2)  
    y_train, y_test = y_train.values.astype('int'), y_test.values.astype('int')


    # initialize the arrays to be return to the main function
    loss_array = []
    acc_array = []
    test_epoch_array = []
    
    if model_type == 'vertical':
    
        num_contiuous, num_categorical = 13, 26
        
        # set up the attribute split scheme for vertical FL
        attribute_split_contiuous = \
            np.ones(organization_num).astype(int) * \
            int(num_contiuous/organization_num)
            
        attribute_split_categorical = \
            np.ones(organization_num).astype(int) * \
            int(num_categorical/organization_num)
            
        # correct the attribute split scheme if the total attribute number is larger than the actual attribute number
        if np.sum(attribute_split_contiuous) > num_contiuous or np.sum(attribute_split_categorical) > num_categorical:
            print('unknown error in attribute splitting!')
        elif np.sum(attribute_split_contiuous) < num_contiuous:
            missing_attribute_num = num_contiuous - np.sum(attribute_split_contiuous)
            attribute_split_contiuous[-1] = attribute_split_contiuous[-1] + missing_attribute_num
        else:
            print('Successful attribute split for multiple organizations')    
        
        if np.sum(attribute_split_categorical) < num_categorical:
            missing_attribute_num = num_categorical - np.sum(attribute_split_categorical)
            attribute_split_categorical[-1] = attribute_split_categorical[-1] + missing_attribute_num
        else:
            print('Successful attribute split for multiple organizations')
    
    
        
        # define the attributes in each group for each organization
        attribute_groups_contiuous, attribute_groups_categorical = [], []
        feature_columns_contiuous = [i for i in range(num_contiuous)]
        feature_columns_categorical = [num_contiuous+i for i in range(num_categorical)]
        attribute_start_contiuous, attribute_start_categorical = 0, 0
        for organization_idx in range(organization_num):
            attribute_end_contiuous = attribute_start_contiuous + attribute_split_contiuous[organization_idx]
            attribute_groups_contiuous.append(feature_columns_contiuous[attribute_start_contiuous : attribute_end_contiuous])
            attribute_start_contiuous = attribute_end_contiuous
            
            attribute_end_categorical = attribute_start_categorical + attribute_split_categorical[organization_idx]
            attribute_groups_categorical.append(feature_columns_categorical[attribute_start_categorical : attribute_end_categorical])
            attribute_start_categorical = attribute_end_categorical
            print('The attributes held by Organization {0}: {1}, {2}'.format(organization_idx, attribute_groups_contiuous[organization_idx], attribute_groups_categorical[organization_idx]))                        

        X_train_vertical_FL, X_test_vertical_FL = {}, {}
        vertical_splitted_feature = {}
        for organization_idx in range(organization_num):
            X_train_vertical_FL[organization_idx] = \
                np.hstack((X_train[:, attribute_groups_contiuous[organization_idx]],\
                           X_train[:, attribute_groups_categorical[organization_idx]]))
            X_test_vertical_FL[organization_idx] = \
                np.hstack((X_test[:, attribute_groups_contiuous[organization_idx]],\
                           X_test[:, attribute_groups_categorical[organization_idx]]))
                  
            vertical_splitted_feature[organization_idx] = \
                [feature_columns[0][attribute_groups_contiuous[organization_idx][0]:attribute_groups_contiuous[organization_idx][-1]+1], \
                feature_columns[1][attribute_groups_categorical[organization_idx][0]-num_contiuous:attribute_groups_categorical[organization_idx][-1]-num_contiuous+1]]
            print('The shape of the encoded dataset held by Organization {0}: {1}'.format(organization_idx, X_train_vertical_FL[organization_idx].shape))                     
            print('Selected features:', vertical_splitted_feature[organization_idx])             
    
     
        organization_hidden_units_array = [np.array([256])]*organization_num
        organization_output_dim = 64
    
        top_hidden_units = [64]
        top_output_dim = 1
    
        activation = 'relu'
            
        # build the client models
        organization_models = {}
        for organization_idx in range(organization_num):
            organization_models[organization_idx] = \
                tf_organization_graph(vertical_splitted_feature[organization_idx],
                    organization_hidden_units_array[organization_idx],
                    organization_output_dim, activation)
    

        top_model = tf_top_graph(k, w_reg, v_reg, top_hidden_units, top_output_dim, activation)
           
        optimizer = optimizers.Adam(
        learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
        # optimizer = optimizers.SGD(learning_rate=0.005, momentum=0.5)
        # optimizer = optimizers.SGD(0.001)
    
        for i in range(epochs):
            
            batch_idxs_list = batch_split(len(X_train[0]), args.batch_size, args.batch_type)
            
            for batch_idxs in batch_idxs_list:
            
                with tf.GradientTape(persistent = True) as tape:
    
                    organization_outputs = {}
                    for organization_idx in range(organization_num):
                        organization_outputs[organization_idx] = \
                            organization_models[organization_idx](X_train_vertical_FL[organization_idx][batch_idxs])                                   
    
                    y_pre = top_model(organization_outputs)
                    
                    loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train[batch_idxs], y_pred=y_pre))
    
                    top_grad = tape.gradient(loss, top_model.variables)
                    
                    optimizer.apply_gradients(grads_and_vars=zip(top_grad, top_model.variables))
            
                    # Kang: the following codes need additional work
                    for organization_idx in range(organization_num):
                        organization_model_grad = tape.gradient(loss, organization_models[organization_idx].variables)     
                        optimizer.apply_gradients(grads_and_vars=zip(organization_model_grad, organization_models[organization_idx].variables))
                    
                    del tape    

            # let the program report the simulation progress
            if (i+1)%1 == 0:
                organization_outputs_for_test = {}
                for organization_idx in range(organization_num):
                    organization_outputs_for_test[organization_idx] = \
                        organization_models[organization_idx](X_test_vertical_FL[organization_idx])
                
                log_probs = top_model(organization_outputs_for_test)
                pre = [1 if x>0.5 else 0 for x in log_probs]
                acc = accuracy_score(y_test, pre)
                print('For the {0}-th epoch, train loss: {1}, test acc: {2}'.format(i+1, loss.numpy(), acc))
                
                test_epoch_array.append(i+1)
                loss_array.append(loss.numpy())
                acc_array.append(acc)

    
    elif model_type == 'centralized':
        
        hidden_units = [256, 128, 64]
        output_dim = 1
        activation = 'relu'
        
        model = DeepFM(feature_columns, k, w_reg, v_reg, hidden_units, output_dim, activation)
        optimizer = optimizers.SGD(0.01)
        #optimizer = optimizers.SGD(learning_rate=0.005, momentum=0.5)
    
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
        for i in range(epochs):
            
            batch_idxs_list = batch_split(len(X_train[0]), args.batch_size, args.batch_type)
            
            for batch_idxs in batch_idxs_list:
            
                with tf.GradientTape() as tape:
                    y_pre = model(X_train[batch_idxs])
                    loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train[batch_idxs], y_pred=y_pre))
                grad = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
    
            #evaluate
            if (i+1)%1 == 0:
                pre = model(X_test)
                pre = [1 if x>0.5 else 0 for x in pre]
                acc = accuracy_score(y_test, pre)
                print('For the {}-th epoch, train loss: {}, test acc: {}'.format(i, loss.numpy(), acc))
    
                test_epoch_array.append(i+1)
                loss_array.append(loss.numpy())
                acc_array.append(acc)
            
    return test_epoch_array, loss_array, acc_array
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vertical FL')
    parser.add_argument('--dname', default='criteo', help='dataset name')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs') 
    parser.add_argument('--batch_type', type=str, default='mini-batch')  
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--data_type', default='original', help='define the data options: original or one-hot encoded')
    parser.add_argument('--model_type', default='vertical', help='define the learning methods: vertical or centralized')    
    parser.add_argument('--organization_num', type=int, default='2', help='number of origanizations, if we use vertical FL')    


    args = parser.parse_args()

    start_time = time.time()
    
    # collect experimental results
    test_epoch_array, loss_array, acc_array = main(args)

    elapsed = time.time() - start_time
    mins, sec = divmod(elapsed, 60)
    hrs, mins = divmod(mins, 60)
    print("Elasepd time: %d:%02d:%02d" % (hrs, mins, sec))
    
    # quickly plot figures to see the loss and acc performance
    figure_loss = plt.figure(figsize = (8, 6))
    plt.plot(test_epoch_array, loss_array)
    plt.xlabel('Communication round #')
    plt.ylabel('Test loss')
    plt.ylim([0, 1])
    
    
    figure_acc = plt.figure(figsize = (8, 6))
    plt.plot(test_epoch_array, acc_array)
    plt.xlabel('Communication round #')
    plt.ylabel('Test accuracy')
    plt.ylim([0, 1])  