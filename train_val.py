import copy
import tqdm
import torch
import time
from collections import defaultdict
import matplotlib.pyplot as plt


def train_model(model,config, dataloader):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    early_stopping_counter=0

    optimizer=torch.optim.Adam(model.parameters(),lr=config['LR'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['EPOCH'], eta_min=0,verbose=False)
    loss_fn=torch.nn.BCEWithLogitsLoss()
  
    for epoch in tqdm.tqdm(range(config['EPOCH'])):
        if early_stopping_counter>=5:
          break
        for phase in ['train', 'val']:      
            if phase == 'train':               
                model.train()  
            else:
                model.eval()  
            metrics = defaultdict(float)
            epoch_samples = 0
          
            for btch,feed_dict in enumerate(dataloader[phase]):
                inputs=feed_dict[0]
                inputs=inputs.unsqueeze(0)
                labels=feed_dict[1]
                
                inputs = inputs.type(torch.FloatTensor)
                inputs = inputs.to(config['device'])
                labels = labels.type(torch.FloatTensor)
                labels = labels.to(config['device'])
                
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  
                    outputs=outputs.squeeze()  
                    loss=loss_fn(outputs,labels)
                    metrics['loss']+=loss.item()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()  
                epoch_samples += 1 
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'val':
                if epoch_loss<best_loss:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_loss=epoch_loss
                    early_stopping_counter=0
                else:
                    early_stopping_counter+=1

        scheduler.step()           
    model.load_state_dict(best_model_wts)       
    return model


def train_ssl(model,config,dataloader):
    train_losses=[]
    val_losses=[]
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Change model's last layer:
    model.output_layer=torch.nn.Linear(config['REPRESENTATION_LAYER'],config['project_dim'])
    model=model.to(config['device'])

    optimizer=torch.optim.Adam(model.parameters(),lr=config['LR'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['EPOCH'], eta_min=0,verbose=False)
    ssl_loss = torch.nn.MSELoss()
    
   
    

    for epoch in tqdm.tqdm(range(config['ssl_epochs'])):
        for phase in ['train', 'val']:      
            if phase == 'train':               
                model.train()  
            else:
                model.eval()  
            metrics = defaultdict(float)
            epoch_samples = 0
          
            for btch,feed_dict in enumerate(dataloader[phase]):
                inputs=feed_dict[0]
              
                inputs = inputs.type(torch.FloatTensor)
                num_elements = int(torch.prod(torch.tensor(inputs.shape)))
                # Determine the number of zeros and ones
                num_zeros = int(num_elements * config['ssl_corruption'])
                num_ones = num_elements - num_zeros
                tensor_zeros = torch.zeros(num_zeros)  # Create a tensor of zeros
                tensor_ones = torch.ones(num_ones)  # Create a tensor of ones

                # Concatenate the tensors of zeros and ones
                tensor_data = torch.cat((tensor_zeros, tensor_ones))
    
                # Shuffle the tensor
                tensor_shuffled = tensor_data[torch.randperm(num_elements)].reshape(inputs.shape)
                to_predict=inputs.detach().clone()
                inputs=tensor_shuffled*inputs
                inputs=inputs.unsqueeze(0)

                inputs = inputs.to(config['device'])
                to_predict = to_predict.to(config['device'])
                to_predict=to_predict.unsqueeze(0)

            
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    
                    predicted = model(inputs)

                    loss=ssl_loss(predicted,to_predict)
                    metrics['loss']+=(loss.item()*inputs.size(0))
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
               
                epoch_samples += inputs.size(0) 
            epoch_loss = metrics['loss'] / epoch_samples
         


          
            if phase == 'val':
                if epoch_loss<best_loss:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_loss=epoch_loss
                    
                val_losses.append(epoch_loss)
            else:
              train_losses.append(epoch_loss)
                   

        scheduler.step()           
    model.load_state_dict(best_model_wts)     
    # Change back to classification layer
    model.output_layer=torch.nn.Linear(config['REPRESENTATION_LAYER'],1)
    model=model.to(config['device'])

    return model
