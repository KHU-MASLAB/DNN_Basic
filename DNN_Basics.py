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
    
    dataloader = DataLoader(TensorDataset(data_input, data_output),
                            batch_size=1024,
                            shuffle=True,
                            pin_memory=False,
                            num_workers=4,
                            persistent_workers=True)
    optimizer = torch.optim.Adam([{"params": DNN.parameters(), "lr": 1e-3}])
    
    network_parameters = []
    loss_values = []
    for epoch in range(Epochs):
        total_data = []
        total_pred = []
        for idx_batch, batch in enumerate(dataloader):
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
            
            # Save unscaled result of one epoch
            with torch.no_grad():
                pred_output = pred_output * output_std + output_mean
                batch_output = batch_output * output_std + output_mean
            total_data.append(batch_output.detach().cpu())
            total_pred.append(pred_output.detach().cpu())
        # Concatenate batches
        total_data = torch.cat(total_data, dim=0)
        total_pred = torch.cat(total_pred, dim=0)
        
        # Calculate accuracies
        R2score = r2_score(total_data, total_pred)
        UnscaledMSE = torch.square(torch.mean(total_data - total_pred))
        print(f"Epoch {epoch + 1}: R2={R2score:.5f}, UnscaledMSE={UnscaledMSE:.5E}")
        
        # Record network parameters and loss values
        network_parameters.append(DNN.state_dict())
        loss_values.append(UnscaledMSE)
    
    # Save network parameters with normalization parameters
    best_network_param = network_parameters[np.argmin(loss_values)]
    network_info = {"input_mean": input_mean, "input_std": input_std,
                    "output_mean": output_mean, "output_std": output_std,
                    "loss_values": loss_values, "state_dict": best_network_param}
    torch.save(network_info, "DNN.pt")
    print(f"Saved the best model from Epoch {np.argmin(loss_values)} (UnscaledMSE={np.min(loss_values):.5E})")

def LoadAndPredict():
    # Initialize MLP module
    Num_hiddenlayers = 4
    Num_neurons = 300
    Activation = 'relu'
    DNN = MLP(len(Input), Num_hiddenlayers, Num_neurons, len(Output), activation=Activation).cuda()
    
    # Load
    network_info = torch.load("DNN.pt")
    DNN.load_state_dict(network_info["state_dict"])
    
    data_input = torch.FloatTensor(TrainData[Input].to_numpy())
    data_output = torch.FloatTensor(TrainData[Output].to_numpy())
    
    input_mean = network_info["input_mean"]
    input_std = network_info["input_std"]
    output_mean = network_info["output_mean"]
    output_std = network_info["output_std"]
    loss_values = network_info["loss_values"]
    
    # Predict
    DNN.eval()
    with torch.no_grad():
        data_input = data_input.cuda()  # to GPU
        data_input = (data_input - input_mean) / input_std  # Scale input
        
        total_pred = DNN.forward(data_input)
        
        data_input = data_input * input_std + input_mean
        data_input = data_input.cpu()  # to CPU
        total_pred = total_pred * output_std + output_mean
        total_pred = total_pred.cpu()  # to CPU
    
    # Plot
    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    ax[0, 0].scatter(data_output, total_pred, s=4, c='black')
    R2score = r2_score(data_output, total_pred)
    ax[0, 0].set_title(f"$R^2$={R2score:.5f}", fontsize=25)
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
    # Train()
    LoadAndPredict()
