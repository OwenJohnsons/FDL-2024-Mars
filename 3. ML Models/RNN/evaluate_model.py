import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch

# supporting file imports
from training_config import *
from get_data import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import torchmetrics

#---------------------------------------------------------------------------------------------------
# load a trained model
model_path = "/home/jupyter/fdl-2024-mars/ML_Models/RNN/trained_models/20240724-185642/model.pt"
model_train.load_state_dict(torch.load(model_path))
model_train.eval()

#---------------------------------------------------------------------------------------------------
# set up figure saving stuff
model_basename = os.path.basename(os.path.dirname(model_path))

#make the directory corresponding to basename
fig_save_dir = os.path.join(model_eval_fig_dir,model_basename)
# os.mkdir(fig_save_dir, exist_ok=)
if not os.path.exists(fig_save_dir):
    os.mkdir(fig_save_dir)

#---------------------------------------------------------------------------------------------------
# LOAD TEST DATA
# test_data = DataLoader(CustomDataset(root_data_dir, test_EID_file), batch_size=batch_size_defined,
#                         shuffle=False, collate_fn=collate_custom)

# Pugazh's change:
test_data = DataLoader(CustomDataset(root_data_dir, test_EID_file, data_exclude_list), batch_size=batch_size_defined,
                        shuffle=False, collate_fn=collate_custom)
#---------------------------------------------------------------------------------------------------
# define all class labels for confusion matrix computation
class_labels = ['carbonate','chloride','iron oxide', 'nitrate', 'OOC',
                'oxychlorine','phyllosilicate','silicate','sulfate','sulfide']

# define the global variables to be used in confusion matrix calculation
pred_global = []
true_global = []

for inputs, label in test_data:
    
    # Pugazh's change:
    inputs, label = inputs.to(device), label.to(device)
    
    # inputs, label = data
    pred = model_train(inputs)

    # apply the softmax layer for probabilities here
    pred_prob = F.softmax(pred,dim=0)

    # append to global arrays
    # pred_global.extend(pred_prob.detach().numpy())
    # true_global.extend(label.detach().numpy())
    
    # Pugazh's change:
    pred_global.extend(pred_prob.cpu().detach().numpy())
    true_global.extend(label.cpu().detach().numpy())

#--------------------------------
# compute confusion matrix
metric1 = MulticlassConfusionMatrix(num_classes=len(class_labels))
metric1.update(torch.tensor(pred_global),torch.tensor(true_global))

plt.rcParams["figure.figsize"] = (20,20)

fig1,ax1 = metric1.plot()
ax1.set_xticklabels(class_labels)
ax1.set_yticklabels(class_labels)
fig1.savefig(os.path.join(fig_save_dir,'ConfusionMatrix.png'),dpi=300, bbox_inches = "tight")

