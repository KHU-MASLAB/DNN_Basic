from Vars import *
from Module_DNN import MLP
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
import torch
import numpy as np
import matplotlib.pyplot as plt

def Train():
    Epochs = 1000
    Num_hiddenlayers = 4
    Num_neurons = 300
    Activation = 'relu'
    DNN = MLP(len(Input), Num_hiddenlayers, Num_neurons, len(Output), activation=Activation).cuda()
    
    data_input = torch.FloatTensor(TrainData[Input].to_numpy())
    data_output = torch.FloatTensor(TrainData[Output].to_numpy())
    
    input_mean = data_input.mean(dim=0).cuda()
    output_mean = data_output.mean(dim=0).cuda()
    input_std = data_input.std(dim=0).cuda()
    output_std = data_output.std(dim=0).cuda()
    
    dataloader_train = DataLoader(TensorDataset(data_input, data_output),
                            batch_size=1024,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4,
                            persistent_workers=True)
    dataloader_eval = DataLoader(TensorDataset(data_input, data_output),
                                  batch_size=1024,
                                  shuffle=False, # affects R2 score
                                  pin_memory=True,
                                  num_workers=2,
                                  persistent_workers=True)
    optimizer = torch.optim.Adam([{"params": DNN.parameters(), "lr": 1e-3}])
    
    network_parameters = []
    loss_values = []
    r2_values = []
    for epoch in range(Epochs):
        total_data = []
        total_pred = []
        # Training stage
        for idx_batch, batch in enumerate(dataloader_train):
            batch_input, batch_output = batch
            
            # Send batch to GPU
            batch_input = batch_input.cuda()
            batch_output = batch_output.cuda()
            
            # Data scaling
            with torch.no_grad():
                batch_input = (batch_input - input_mean) / input_std
                batch_output = (batch_output - output_mean) / output_std
            
            # Forward
            pred_output = DNN.forward(batch_input)
            
            # Compute loss
            loss = torch.mean(torch.square(pred_output - batch_output))
            
            # Backward
            for param in DNN.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()
        # End of training stage
        
        # Evaluation stage
        DNN.eval()
        with torch.no_grad():
            for idx_batch, batch in enumerate(dataloader_eval):
                batch_input, batch_output = batch
        
                # Send batch to GPU
                batch_input = batch_input.cuda()
        
                # Data scaling
                with torch.no_grad():
                    batch_input = (batch_input - input_mean) / input_std
        
                # Forward
                pred_output = DNN.forward(batch_input)
                
                # Unscale
                pred_output = pred_output * output_std + output_mean
                
                # Save
                total_data.append(batch_output)
                total_pred.append(pred_output.cpu())
        DNN.train()
        # End of evaluation stage
        
        # Concatenate batches
        total_data = torch.cat(total_data, dim=0).flatten()
        total_pred = torch.cat(total_pred, dim=0).flatten()
        
        # Compute accuracy
        R2score = r2_score(total_data, total_pred)
        UnscaledMSE = torch.mean(torch.square(total_data - total_pred))
        print(f"Epoch {epoch + 1}: R2={R2score:.4f}, UnscaledMSE={UnscaledMSE:.4E}")
        
        # Record network parameters
        current_network_parameter=DNN.state_dict()
        for name,param in current_network_parameter.items():
            current_network_parameter[name]=param.cpu() # Params must be saved on CPU
        network_parameters.append(current_network_parameter)
        
        # Record accuracy scores
        loss_values.append(UnscaledMSE)
        r2_values.append(R2score)
    
    # Save network parameters and normalization parameters (mean, std of input, output)
    best_network_param = network_parameters[np.argmin(loss_values)]
    network_info = {"input_mean": input_mean, "input_std": input_std,
                    "output_mean": output_mean, "output_std": output_std,
                    "loss_values": loss_values, "state_dict": best_network_param}
    torch.save(network_info, "DNN.pt")
    print(f"Saved the best model from Epoch {np.argmin(loss_values) + 1} (UnscaledMSE={np.min(loss_values):.4E})")

def LoadAndPredict():
    # Initialize MLP module
    Num_hiddenlayers = 4
    Num_neurons = 300
    Activation = 'relu'
    DNN = MLP(len(Input), Num_hiddenlayers, Num_neurons, len(Output), activation=Activation).cuda()
    
    # Load
    network_info = torch.load("DNN.pt")
    DNN.load_state_dict(network_info["state_dict"])
    
    input_mean = network_info["input_mean"]
    input_std = network_info["input_std"]
    output_mean = network_info["output_mean"]
    output_std = network_info["output_std"]
    loss_values = network_info["loss_values"]

    data_input = torch.FloatTensor(TrainData[Input].to_numpy())
    data_output = torch.FloatTensor(TrainData[Output].to_numpy())
    dataloader_eval = DataLoader(TensorDataset(data_input, data_output),
                                  batch_size=1024,
                                  shuffle=False, # affects R2 score
                                  pin_memory=True,
                                  num_workers=2,
                                  persistent_workers=True)
    # Predict
    DNN.eval()
    total_data=[]
    total_pred=[]
    with torch.no_grad():
        for idx_batch, batch in enumerate(dataloader_eval):
            batch_input, batch_output = batch
        
            # Send batch to GPU
            batch_input = batch_input.cuda()
        
            # Data scaling
            with torch.no_grad():
                batch_input = (batch_input - input_mean) / input_std
        
            # Forward
            pred_output = DNN.forward(batch_input)
        
            # Unscale
            pred_output = pred_output * output_std + output_mean
            total_data.append(batch_output)
            total_pred.append(pred_output.cpu())

    total_data = torch.cat(total_data, dim=0).flatten()
    total_pred = torch.cat(total_pred, dim=0).flatten()
    R2score = r2_score(data_output, total_pred)
    UnscaledMSE = torch.mean(torch.square(total_data - total_pred))
    
    # Plot
    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    ax[0, 0].scatter(data_output, total_pred, s=4, c='black')
    ax[0, 0].set_title(f"$R^2$={R2score:.4f}\nUnscaledMSE={UnscaledMSE:.4E}", fontsize=18)
    ax[0, 0].set_xlabel("Label", fontsize=20)
    ax[0, 0].set_ylabel("Prediction", fontsize=20)
    
    ax[0, 1].plot(range(len(loss_values)), loss_values, color='green', linewidth=1)
    # ax[0, 1].set_yscale('linear')
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_xlabel("Training epochs", fontsize=20)
    ax[0, 1].set_ylabel("Unscaled MSE of y", fontsize=20)
    
    ax[1, 0].scatter(data_input[:, 0], data_output.flatten(), s=2, c='grey', label='Label')
    ax[1, 0].scatter(data_input[:, 0], total_pred.flatten(), s=2, c='red', label='Prediction')
    ax[1, 0].set_xlabel("x1", fontsize=20)
    ax[1, 0].set_ylabel("y", fontsize=20)
    ax[1, 0].legend(loc=2)
    
    ax[1, 1].scatter(data_input[:, 1], data_output.flatten(), s=2, c='grey', label='Label')
    ax[1, 1].scatter(data_input[:, 1], total_pred.flatten(), s=2, c='red', label='Prediction')
    ax[1, 1].set_xlabel("x2", fontsize=20)
    ax[1, 1].set_ylabel("y", fontsize=20)
    ax[1, 1].legend(loc=2)
    
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    Train()
    LoadAndPredict()