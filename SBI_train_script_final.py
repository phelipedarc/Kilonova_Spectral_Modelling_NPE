#Gpu livre
GPU = input("Gpu livre: ")
#Modelo mdn, maf, mab ,nsf
MODEL = input("Modelo mdn, maf, mab ,nsf:  ")
#SIGMA DO RUIDO GAUSSIANO
RUIDO = int(input("SIGMA DO RUIDO GAUSSIANO 5:  "))
#SIGMA DO SMOOTH:
SMOOTH = int(input("Smooth on the Kn sim 2:  "))
#NUMERO DE SIMULAÇÕES:
NUM_SIM = int(input("NUMERO DE SIMULAÇÕES 100000:  "))
# Adding the noise coming from the real data:
noisekn = float(input("Adding the noise of the Kn;: 1 (yes) and 0 (no):  "))








import os, sys
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
from torch import nn
import random
import scipy
import kilonovanet
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm.notebook import tqdm
from scipy import stats as st
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import prepare_for_sbi, simulate_for_sbi, SNLE, ratio_estimator_based_potential, SNRE_A,SNRE,SNPE
from sbi.inference import likelihood_estimator_based_potential, ImportanceSamplingPosterior, MCMCPosterior
import torch
from scipy.stats import binned_statistic, linregress, gaussian_kde
import pandas as pd
from sbi.utils.get_nn_models import posterior_nn
import pickle
import math


#/dados10Tdock2/phelipedata/Kilonova_Simulation_Based_Inference/Notebooks/SCRIPTS/KFOLD/X_data_noise_5_realgwnoise_1.0_numsamples_100000.npy
def mkdir(directory_path): 
    #cria as pastas dado o path e retorna o path
    if os.path.exists(directory_path): 
        return directory_path
    else: 
        try: 
            os.makedirs(directory_path)
        except: 
            # in case another machine created the path meanwhile !:(
            return sys.exit("Erro ao criar diretório") 
        return directory_path


def flux_smoothed(arr, sigma):
    pc_to_cm = 3.086e18
    #flux_real = arr / (10*pc_to_cm) ** 2
    flux_real = arr*(1/(4*math.pi*(10*3.086e18)**2))
    smooth = gaussian_filter(flux_real, sigma, 0)
    return smooth

# def simulate_spec(params):
#     wl = np.linspace(100.0, 99900, 500)
#     time = [1.5,2.5,3.5]
#     spec = model.predict_spectra(params,time)
#     smoothed_flux0 = flux_smoothed(spec[0][0], float(SMOOTH))
#     smoothed_flux1 = flux_smoothed(spec[0][1], float(SMOOTH))
#     smoothed_flux2 = flux_smoothed(spec[0][2], float(SMOOTH))

#     smoothed_sed0 = np.vstack([wl, smoothed_flux0]).T
#     flux_interp0 = np.interp(wl_sim, smoothed_sed0[:,0], smoothed_sed0[:,1])
    
#     smoothed_sed1 = np.vstack([wl, smoothed_flux1]).T
#     flux_interp1 = np.interp(wl_sim, smoothed_sed1[:,0], smoothed_sed1[:,1])
    
#     smoothed_sed2 = np.vstack([wl, smoothed_flux2]).T
#     flux_interp2 = np.interp(wl_sim, smoothed_sed2[:,0], smoothed_sed2[:,1])
    
#     flux_interp = np.concatenate((flux_interp0.reshape((550,1)),flux_interp1.reshape((550,1)),flux_interp2.reshape((550,1))),axis=-1)
#     idx_noise = [5,6,4,2,1,0,7]
#     for i in range(len(flux_interp)):
#         for j in range(flux_interp.shape[-1]):
#             noise_std = (5/100)*flux_interp[i,j]
#             error = np.random.normal(0, noise_std, 1)
#             error2idx = random.choice(idx_noise)
#             flux_interp[i,j] = flux_interp[i,j] + error + (knreal2[error2idx,:,1] -scipy.ndimage.uniform_filter1d(knreal2[error2idx,:,1],100))[i]
    
#     return flux_interp
    
def simulate_spec(params):
    '''Simulator that takes as input the Dietrich based simulator parameters and
    Returns a Kilonova spectra ranging from 5000A to 8000A
    
    - Smoothing on the Kilonova raw spectra to reduce the noise coming from the kilonovanet simulator
    
    - Added a Gaussian noise
    
    - Adding a noise from the real spectra --> noise_kn = (spectra - smooth(spectra))
    
    '''
    wl = np.linspace(100.0, 99900, 500)
    time = [1.5,2.5,3.5]
    spec = model.predict_spectra(params,time)
    smoothed_flux0 = flux_smoothed(spec[0][0], SMOOTH)
    smoothed_flux1 = flux_smoothed(spec[0][1], SMOOTH)
    smoothed_flux2 = flux_smoothed(spec[0][2], SMOOTH)

    smoothed_sed0 = np.vstack([wl, smoothed_flux0]).T
    flux_interp0 = np.interp(wl_sim, smoothed_sed0[:,0], smoothed_sed0[:,1])
    
    smoothed_sed1 = np.vstack([wl, smoothed_flux1]).T
    flux_interp1 = np.interp(wl_sim, smoothed_sed1[:,0], smoothed_sed1[:,1])
    
    smoothed_sed2 = np.vstack([wl, smoothed_flux2]).T
    flux_interp2 = np.interp(wl_sim, smoothed_sed2[:,0], smoothed_sed2[:,1])
    
    flux_interp = np.concatenate((flux_interp0.reshape((550,1)),flux_interp1.reshape((550,1)),flux_interp2.reshape((550,1))),axis=-1)
    #applying the noise:
    idx_noise = [5,6,4,2,1,0,7]
    for i in range(len(flux_interp)):
        for j in range(flux_interp.shape[-1]):
            noise_std = (RUIDO/100)*flux_interp[i,j]
            error = np.random.normal(0, noise_std, 1)
            error2idx = random.choice(idx_noise)
            flux_interp[i,j] = flux_interp[i,j] + error 
    
    return flux_interp    
    
    
    
    
# class Model5(nn.Module):
#     def __init__(self,filters1,filters2,filters3,ks1,ks2,ks3,lstm1,lstm2,outfeat):
#         super().__init__()
        
#         #num filtro e tipos de funções de ativação
#         self.conv1 = nn.Conv1d(in_channels=3, out_channels=filters1, kernel_size=2,padding='same')
#         self.act1 = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(kernel_size=ks1, stride=ks1)
#         #divide por 2
#         self.conv2 = nn.Conv1d(in_channels=filters1, out_channels=filters2, kernel_size=2,padding='same')
#         self.act2 = nn.ReLU()
#         self.pool2 = nn.MaxPool1d(kernel_size=ks2, stride=ks2)
#         #divide por 3
#         self.conv3 = nn.Conv1d(in_channels=filters2, out_channels=filters3, kernel_size=2,padding='same')
#         self.act3 = nn.ReLU()
#         self.pool3 = nn.MaxPool1d(kernel_size=ks3, stride=ks3)
#         #divide por 2
#         #550/2*2*3
#         self.lstm = nn.LSTM(input_size=filters3, hidden_size=lstm1, bidirectional=True, batch_first=True)
#         self.lstm2 = nn.LSTM(input_size=lstm1*2, hidden_size=lstm2, bidirectional=True, batch_first=True)
#         # in features do fc1 = 550/2*3*2 )* 256*2
#         # 68
#         self.flat = nn.Flatten()
#         self.fc1 = nn.Linear(in_features=int((lstm2*2)*(int((550)/(ks1*ks2*ks3)))), out_features=outfeat)
#         self.act4 = nn.ReLU()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.act1(x)
#         x = self.pool1(x)
#         #x = self.dropout1(x)
#         x = self.conv2(x)
#         x = self.act2(x)
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = self.act3(x)
#         x = self.pool3(x)
#         x = torch.permute(x, (0,2,1)) 
#         x, _ = self.lstm(x)
#         x, _ = self.lstm2(x)
#         x = self.flat(x)
#         x = self.fc1(x)
#         x = self.act4(x)
#         return x
class Model5(nn.Module):
    def __init__(self,filters1,filters2,filters3,ks1,ks2,ks3,lstm1,lstm2,outfeat):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=filters1, kernel_size=2,padding='same')
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=ks1, stride=ks1)
        #divide por 2
        #self.dropout1 = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv1d(in_channels=filters1, out_channels=filters2, kernel_size=2,padding='same')
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=ks2, stride=ks2)
        #divide por 3
        self.conv3 = nn.Conv1d(in_channels=filters2, out_channels=filters3, kernel_size=2,padding='same')
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=ks3, stride=ks3)
        #divide por 2
        #550/2*2*3
        self.lstm = nn.LSTM(input_size=filters3, hidden_size=lstm1, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstm1*2, hidden_size=lstm2, bidirectional=True, batch_first=True)
        # in features do fc1 = 550/2*3*2 )* 256*2
        # 68
        self.flat = nn.Flatten()
        #Dropout
        #self.dropout2 = nn.Dropout(p=0.20)
        #
        self.fc1 = nn.Linear(in_features=int((lstm2*2)*(int((550)/(ks1*ks2*ks3)))), out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=outfeat)
        self.act4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        #x = self.dropout1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)
        x = torch.permute(x, (0,2,1)) 
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)
        x = self.flat(x)
        #x = self.dropout2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.act4(x)
        return x


#---------EMBEBEDING NET PARAMETERS--------------------------------------
testefilters1=128
testefilters2=128
testefilters3=64
testeks1= 2
testeks2= 2
testeks3= 3
testelstm1= 300
testelstm2= 300
testeoutfeat = 100
#------------------TRAINING PARAMETERS------------------------------------
batch_size=2048
stop_epoch=80
max_epochs=1000
combined_loss=False
retrain_from_scratch1=True
#--------------------DENSITY ESTIMATOR PARAMETERS---------------------------------------
num_bins=20 
num_transforms=4
hidden_features=5
#---------------------------------------------------------------------------------------
metadata = '/tf/dados10Tdock2/phelipedata/Kilonovas-spec/Pytorch_notebooks/data_kilonovanet/metadata_bulla_bns.json'
model_file = '/tf/dados10Tdock2/phelipedata/Kilonovas-spec/Pytorch_notebooks/data_kilonovanet/bulla-bns-latent-20-hidden-1000-CV-4-2021-04-21-epoch-200.pt'
knreal2 =np.load('/tf/dados10Tdock2/phelipedata/KilonovasDataset/NEWKNrealpreprocessed2nonorm.npy',allow_pickle=True)
#---------------------------------------------------------------------------------------
model = kilonovanet.Model(metadata, model_file)
prior = utils.BoxUniform(low = torch.tensor([1e-3, 1e-2, 0.0, 0.0]), 
                         high = torch.tensor([2e-2, 0.13, 90, 1.0]))
prior_cuda = utils.BoxUniform(low = torch.tensor([1e-3, 1e-2, 0.0, 0.0]), 
                         high = torch.tensor([2e-2, 0.13, 90, 1.0]), device='cuda')
wl_sim = np.linspace(5000.0, 8000.0, 550)
print('Range of the Spectra: 5000 - 8000')
simulator, prior = prepare_for_sbi(simulate_spec, prior)

from pathlib import Path
#dados10Tdock2/phelipedata/Kilonova_Simulation_Based_Inference/Article/ANPE/Deep_Ensemble_Models
path_data = Path(f"/tf/dados10Tdock2/phelipedata/Kilonova_Simulation_Based_Inference/Article/ANPE/Deep_Ensemble_Models/KFOLD/X_data_noise_"+str(RUIDO)+'_realgwnoise_'+str(noisekn)+'_numsamples_'+str(NUM_SIM)+'.npy')
if path_data.is_file():
    print("File exists")
    X_datakilonet = np.load(f"/tf/dados10Tdock2/phelipedata/Kilonova_Simulation_Based_Inference/Article/ANPE/Deep_Ensemble_Models/KFOLD/X_data_noise_"+str(RUIDO)+'_realgwnoise_'+str(noisekn)+'_numsamples_'+str(NUM_SIM)+'.npy')
    theta = np.load(f"/tf/dados10Tdock2/phelipedata/Kilonova_Simulation_Based_Inference/Article/ANPE/Deep_Ensemble_Models/KFOLD/y_data_noise_"+str(RUIDO)+'_realgwnoise_'+str(noisekn)+'_numsamples_'+str(NUM_SIM)+'.npy')
    print('Data Loaded')

else:
    print("File does not exist")
    theta, X_datakilonet = simulate_for_sbi(simulator, proposal=prior, num_simulations=NUM_SIM)
    X_datakilonet = np.array(X_datakilonet).astype(float)
    theta = np.array(theta).astype(float)
    np.save(f"/tf/dados10Tdock2/phelipedata/Kilonova_Simulation_Based_Inference/Article/ANPE/Deep_Ensemble_Models/KFOLD/X_data_noise_"+str(RUIDO)+'_realgwnoise_'+str(noisekn)+'_numsamples_'+str(NUM_SIM),X_datakilonet)
    np.save(f"/tf/dados10Tdock2/phelipedata/Kilonova_Simulation_Based_Inference/Article/ANPE/Deep_Ensemble_Models/KFOLD/y_data_noise_"+str(RUIDO)+'_realgwnoise_'+str(noisekn)+'_numsamples_'+str(NUM_SIM),theta)



# for i in range(len(X_datakilonet)):
#     for j in range(3):
#         mean= np.mean(X_datakilonet[i][:,j])
#         std = np.std(X_datakilonet[i][:,j])
#         X_datakilonet[i][:,j] = (X_datakilonet[i][:,j]-mean)/abs(std)

print(X_datakilonet.shape)

#Embaralhando o dataset:
theta = np.array(theta).astype(float)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_datakilonet, theta, test_size=0.05, random_state=50)
X_train = np.swapaxes(X_train[:,:,:],1,2)
X_test = np.swapaxes(X_test[:,:,:],1,2)
X_train[0][0].max(),X_train[0][0].min()


from sklearn.model_selection import KFold
# Define the number of folds
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
# Initialize a list to store the results for each fold
results = []
loss_fold=[]
valid_fold=[]
# Perform 5-fold cross-validation
for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    embed_net = Model5(testefilters1,testefilters2,testefilters3,testeks1,testeks2,testeks3,testelstm1,testelstm2,testeoutfeat)
    density_estimator_build_fn = posterior_nn(model=MODEL,embedding_net=embed_net,num_bins=num_bins,num_transforms =num_transforms,hidden_features=hidden_features)
    anpe =SNPE(prior=prior_cuda, density_estimator=density_estimator_build_fn,device='cuda')

    anpe.append_simulations(
        torch.tensor(y_train_fold.astype(np.float32)).to('cuda'),
        torch.tensor(X_train_fold.astype(np.float32)).to('cuda')
    )
    p_x_y_estimator = anpe.train(training_batch_size=batch_size,stop_after_epochs=stop_epoch, max_num_epochs=max_epochs,use_combined_loss=True,
                                 show_train_summary=True,retrain_from_scratch=retrain_from_scratch1)
    hatp_x_y = anpe.build_posterior(p_x_y_estimator)
    #saving the loss and validation:
    loss_fold.append(np.array(anpe._summary['training_log_probs']))
    valid_fold.append(np.array(anpe._summary['validation_log_probs']))
    
    # Save the model for this fold
    #dados10Tdock2/phelipedata/KNfits/Notebooks/SCRIPTS/KFOLD
    #---> Usar o mkdir --->
    model_filename = f"/tf/dados10Tdock2/phelipedata/Kilonova_Simulation_Based_Inference/Article/ANPE/Deep_Ensemble_Models/testmodel_nonorm_fold_{fold}_"+str(MODEL)+"_noise_"+str(RUIDO)+".pkl"
    with open(model_filename, "wb") as handle:
        pickle.dump(hatp_x_y, handle)
        
loss_fold =np.array(loss_fold,dtype=object)  
valid_fold = np.array(valid_fold,dtype=object)
np.save(f"/tf/dados10Tdock2/phelipedata/Kilonova_Simulation_Based_Inference/Article/ANPE/Deep_Ensemble_Models/testvalidation_std_fold_"+str(MODEL)+"_noise_"+str(RUIDO),valid_fold)
np.save(f"/tf/dados10Tdock2/phelipedata/Kilonova_Simulation_Based_Inference/Article/ANPE/Deep_Ensemble_Models/testtrain_std_fold_"+str(MODEL)+"_noise_"+str(RUIDO),loss_fold)
