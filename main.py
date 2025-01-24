#==================================================================
# LSTM Time-Series Classification of Inhalation/Exhalation Periods

# Created: 2025/01/15, Leksi Chen
#==================================================================

from sklearn import metrics
from model import LSTMModel

import matplotlib.pyplot as plt
import preprocessing as pp
import config as cfg

import torch.nn as nn
import torch

import numpy as np
import math
import time
import sys

#====================================
# DATA EXTRACTION AND TRANSFORMATION
#====================================

print('Extracting data...')  # If .mat files need to be processed
if cfg.PREPROCESS_DATA:
    pp.prepare_data(cfg.wave_info, cfg.mask_info, cfg.seq_len, 
                    cfg.pct_train, cfg.mini_batch_size, cfg.ONE_BASED_IND)
    sys.exit()

tensors = torch.load('inhale_exhale_tensor_dat.pt')
train_dataloader, test_dataloader = pp.create_dataloader(tensors[0], tensors[1], 
                                                             tensors[2], tensors[3], 
                                                             cfg.mini_batch_size)
print('Data successfully loaded.')

#=====================
# TRAINING LSTM MODEL
#=====================

# Getting user input to begin training
train_flag = input('Train model? Enter Y or N\n   ')
while train_flag != 'Y' and train_flag != 'N':   train_flag = input('Invalid input. Enter Y or N\n   ')

if train_flag == 'N':
    print('"N" entered. No training will occur. Loading pre-trained neural network instead.')
    model_name = input("Input filename of model parameters to load:\n   ")
    
    model = LSTMModel(cfg.input_size, cfg.hidden_size, cfg.num_layers)
    try: model.load_state_dict(torch.load(model_name, weights_only = True))
    except: 
        print('No such file found.')
        sys.exit()
    loss_fnc = nn.CrossEntropyLoss()
else:
    print('"Y" entered. Commencing training.')
    start_time = time.time()
    
    # Initialization and Parameter Declaration
    model = LSTMModel(cfg.input_size, cfg.hidden_size, cfg.num_layers)
    model = model.to(cfg.device)  # use gpu if available, else use cpu
    loss_fnc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    num_mini_batches = cfg.epochs * math.ceil(np.shape(tensors[0])[0] / cfg.mini_batch_size)  # num training mini-batches
    train_accuracy = [0] * num_mini_batches
    train_loss     = [0] * num_mini_batches
    indices        = [0] * num_mini_batches
    index = 0

    # Training
    for epoch in range(cfg.epochs):
        for batch_x, batch_y in train_dataloader:  # Iterate through mini batches

            optimizer.zero_grad()  # reset gradient
            output = model(batch_x)  # result from LSTM forward pass with batch_x
    
            batch_y = batch_y.view(-1, cfg.seq_len)
            loss = loss_fnc(output.permute(0, 2, 1), batch_y)  # permute for proper argument format

            _, predictions = torch.max(output, 2)  # take max probability for each time sample
            correct_pred = (predictions == batch_y).sum().item()  # adding up correct amounts, cast to int

            # storing accuracy and loss for current mini-batch
            train_accuracy[index] = 100 * correct_pred / (cfg.mini_batch_size * cfg.seq_len)
            train_loss[index] = float(loss)
            indices[index] = index

            loss.backward()
            optimizer.step()  # backward pass and update weights
            index += 1
    
    #================================================
    # GENERATE ACCURACY AND LOSS GRAPHS FOR TRAINING
    #================================================

    # Plot accuracy
    plt.plot(indices, train_accuracy, '-b', marker = '.')
    plt.xlabel('Mini Batch Number', fontdict = cfg.axes_font)
    plt.ylabel('Accuracy (%)', fontdict = cfg.axes_font)
    plt.title('Training Accuracy Per Mini-Batch', fontdict = cfg.title_font)
    plt.savefig('inhale_exhale_classif_training_accuracy.png', dpi = 400)
    plt.close()
    print("Training accuracy plotted and saved to 'inhale_exhale_classif_training_accuracy.png'")
    
    # Plot loss
    plt.plot(indices, train_loss, '-b', marker = '.')
    plt.xlabel('Mini Batch Number', fontdict = cfg.axes_font)
    plt.ylabel('Loss', fontdict = cfg.axes_font)
    plt.title('Loss Per Mini-Batch', fontdict = cfg.title_font)
    plt.savefig('inhale_exhale_classif_training_loss.png', dpi = 400)
    plt.close()
    print("Training loss plotted and saved to 'inhale_exhale_classif_training_loss.png'")
    end_time = time.time()
    print('Training completed. Number of epochs: %d. Number of batches: %d' % (cfg.epochs, index))
    print('Elapsed time: %.3f s' % (end_time - start_time))
    
    # Saving the model
    save_flag = input('Save model parameters? Enter Y or N\n   ')
    while save_flag != 'Y' and save_flag != 'N':    save_flag = input('Invalid input. Enter Y or N\n   ')
    if save_flag == 'N':                            print('"N" entered. Model parameters will not be saved.')
    else:
        try: 
            export_name = input("Input export filename with extension .pt:\n   ")
            torch.save(model.state_dict(), export_name)
            print("Model saved successfully to %s" % export_name)
        except Exception as e:
            print('An error occured: %s' % e)

#====================================
# EVALUATING PERFORMANCE ON TEST SET
#====================================

# Getting user input to begin testing
test_flag = input('Test model? Enter Y or N\n   ')
while test_flag != 'Y' and test_flag != 'N':
    test_flag = input('Invalid input. Enter Y or N\n   ')
if test_flag == 'N':
    print('"N" entered. The model will not be tested.')
else:
    print('"Y" entered. The model will be evaluated on the testing data.')
    model.eval()  # model set to evaluation mode, no weight updates

    num_test_seq = np.shape(tensors[2])[0]  # x_test = tensors[2]
    num_mini_batches = math.ceil(num_test_seq / cfg.mini_batch_size)
    test_accuracies = [0] * num_mini_batches
    test_losses = [0] * num_mini_batches
    test_indices = [0] * num_mini_batches

    model_pred = np.zeros((num_test_seq, cfg.seq_len))

    with torch.no_grad():  # disable gradient calculation for quicker testing
        for batch_num, (batch_x, batch_y) in enumerate(test_dataloader):
            output = model(batch_x)
    
            batch_y = batch_y.view(-1, cfg.seq_len)
            loss = loss_fnc(output.permute(0, 2, 1), batch_y)

            _, predictions = torch.max(output, 2) 
            correct_pred = (predictions == batch_y).sum().item()

            # storing accuracy and loss for current mini-batch
            test_accuracies[batch_num] = 100 * correct_pred / (cfg.mini_batch_size*10)
            test_losses[batch_num] = float(loss)
            test_indices[batch_num] = batch_num

            # storing model predictions
            start = batch_num * cfg.mini_batch_size
            model_pred[start : start + cfg.mini_batch_size][:] = predictions[:]  # second index > length handled automatically


    # Plot accuracy
    plt.plot(test_indices, test_accuracies, '-b', marker = '.')
    plt.xlabel('Mini Batch Number', fontdict = cfg.axes_font)
    plt.ylabel('Accuracy (%)', fontdict = cfg.axes_font)
    plt.title('Training Accuracy Per Mini-Batch', fontdict = cfg.title_font)
    plt.savefig('inhale_exhale_classif_test_accuracy.png', dpi = 400)
    plt.close()
    print("Test accuracy plotted and saved to 'inhale_exhale_classif_test_accuracy.png'")

    # Plot loss
    plt.plot(test_indices, test_losses, '-b', marker = '.')
    plt.xlabel('Mini Batch Number', fontdict = cfg.axes_font)
    plt.ylabel('Loss', fontdict = cfg.axes_font)
    plt.title('Loss Per Mini-Batch', fontdict = cfg.title_font)
    plt.savefig('inhale_exhale_classif_test_loss.png', dpi = 400)
    plt.close()
    print("Test loss plotted and saved to 'inhale_exhale_classif_test_accuracy.png'")

    # Plot confusion matrix
    y_test = np.array(tensors[3]).reshape(-1, 1)  # reshape y_test
    confusion_matrix = 100 * metrics.confusion_matrix(y_test, model_pred.reshape(-1, 1)) / len(y_test)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels = cfg.label_dict)
    cm_display.plot(cmap = plt.cm.Blues)
    plt.title('Normalized Confusion Matrix for Test Data (%)', fontdict=cfg.title_font)
    plt.savefig('inhale_exhale_classif_confusion_matrix.png', dpi = 400)
    plt.close()
    print("Confusion matrix saved to 'inhale_exhale_classif_confusion_matrix.png \nProcess finished.")