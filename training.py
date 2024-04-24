import torch
from tqdm import tqdm
import numpy as np
from torcheval.metrics.text import Perplexity


from visualization import plot_training_curves


# compute the perplexity of prediction and ground truth
def perplexity(pred, target):
    metric = Perplexity()
    metric.update(pred.to('cpu'), target.to('cpu'))
    return metric.compute()


@torch.no_grad()
def feedforward(model, dataloader, device):
    model.eval()
    
    perplexities = []
    losses = []
    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        logits3d, logits, loss = model(x, y)
        perplexities.append(perplexity(logits3d, y).item())
        losses.append(loss.item())    
    # calculate the mean
    perplexities = np.mean(np.array(perplexities))
    losses = np.mean(np.array(losses))
    return losses, perplexities



# back propagation with gradient updates
def backpropagation(model, dataloader, device, optimizer):
    model.train()
    
    perplexities = []
    losses = []
    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        
        logits3d, logits, loss = model(x, y)
        perplexities.append(perplexity(logits3d, y).item())
        losses.append(loss.item()) 
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    # calculate the mean
    perplexities = np.mean(np.array(perplexities))
    losses = np.mean(np.array(losses))
    return losses, perplexities
    
    
# model training loop
def model_training(model, train_loader, valid_loader, device):
    learning_rate = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    n_epochs = 500
    
    # get the initial statistics
    train_loss, train_perplexity = feedforward(model, train_loader, device)
    valid_loss, valid_perplexity = feedforward(model, valid_loader, device)
    print(f"Epoch 0/{n_epochs} | Train Perplexity: {train_perplexity:.3f} | Train Loss: {train_loss:.3f} | Valid Perplexity: {valid_perplexity:.3f} | Valid Loss: {valid_loss:.3f}")
    
    # training curves
    train_losses = [train_loss]
    train_perplexities = [train_perplexity]
    valid_losses = [valid_loss]
    valid_perplexities = [valid_perplexity]
    
    # Early Stopping criteria
    patience = 3
    not_improved = 0
    best_valid_loss = valid_loss
    threshold = 0.01
    
    # training epoches
    for epoch in range(n_epochs):
        # feedforward to estimate loss
        train_loss, train_perplexity = backpropagation(model, train_loader, device, optimizer)
        valid_loss, valid_perplexity = feedforward(model, valid_loader, device)
        
        train_losses.append(train_loss)
        train_perplexities.append(train_perplexity)
        valid_losses.append(valid_loss)
        valid_perplexities.append(valid_perplexity)
        
        print(f"Epoch {epoch+1}/{n_epochs} | Train Perplexity: {train_perplexity:.3f} | Train Loss: {train_loss:.3f} | Valid Perplexity: {valid_perplexity:.3f} | Valid Loss: {valid_loss:.3f}")
        
        # evaluate the current performance
        # strictly better
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            not_improved = 0
            # save the best model based on validation loss
            torch.save(model.state_dict(), f'{type(model).__name__}.pth')
            # also save the optimizer state for future training
            torch.save(optimizer.state_dict(), f'{type(model).__name__}_optimizer.pth')

        # becomes worse
        elif valid_loss > best_valid_loss + threshold:
            not_improved += 1
            if not_improved >= patience:
                print('Early Stopping Activated')
                break
            
    plot_training_curves(train_perplexities, train_loss, valid_perplexities, valid_loss)
    
    
        
        
        