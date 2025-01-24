import torch

# Preprocessing arguments
wave_info = ('waves.mat', 'waves')
mask_info = ('masks.mat', 'masks')
ONE_BASED_IND = True
PREPROCESS_DATA = False

seq_len = int(60 * 44.1 * 1000)
pct_train = 70
mini_batch_size = 2


# Model parameters
input_size = 1
hidden_size = 200
num_layers = 1  # 1 LSTM layer only

epochs = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

axes_font = {'fontname':'Helvetica Neue', 'color':'black', 'size':12}
title_font = {'fontname':'Helvetica Neue', 'color':'black', 'fontweight':'bold', 'size':16}

label_dict = ['Exhale', 'Inhale', 'None']