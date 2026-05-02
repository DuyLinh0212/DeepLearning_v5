import torch
from torch.cuda.amp import autocast
from sklearn import metrics
import numpy as np

def _get_trainable_params(model):
    """Get Parameters with `requires.grad` set to `True`"""
    trainable_params = []
    for x in model.parameters():
        if x.requires_grad:
            trainable_params.append(x)
    return trainable_params

def _evaluate_model(
    model,
    val_loader,
    criterion,
    epoch,
    num_epochs,
    writer,
    current_lr,
    log_every=20,
    device='cpu',
    use_amp=False,
):
    """Runs model over val dataset and returns auc and avg val loss"""

    # Set to eval mode
    model.eval()
    # List of probabilities obtained from the model
    y_probs = []
    # List of groundtruth labels
    y_gt = []
    # List of losses obtained
    losses = []

    # Iterate over the validation dataset
    for i, batch in enumerate(val_loader):
        if batch is None:
            continue
        images, label = batch
        # If GPU is available, load the images and label
        # on GPU
        if device != 'cpu':
            images = [image.to(device, non_blocking=True) for image in images]
            label = label.to(device, non_blocking=True)

        # Obtain the model output by passing the images as input
        with autocast(enabled=bool(use_amp and device != 'cpu')):
            output = model(images)
        # Evaluate the loss by comparing the output and groundtruth label
        loss = criterion(output, label)
        # Add loss to the list of losses
        loss_value = loss.item()
        losses.append(loss_value)
        # Find probability for each class by applying
        # sigmoid function on model output
        probas = torch.sigmoid(output)
        # Add the groundtruth to the list of groundtruths
        y_gt.extend(label.detach().cpu().view(-1).numpy().astype(int).tolist())
        # Add predicted probabilities to the list
        y_probs.extend(probas.detach().cpu().view(-1).numpy().tolist())

        try:
            # Evaluate area under ROC curve based on the groundtruth label
            # and predicted probability
            auc = metrics.roc_auc_score(y_gt, y_probs)
        except:
            # Default area under ROC curve
            auc = 0.5
        # Add information to the writer about validation loss and Area under ROC curve
        writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC', auc, epoch * len(val_loader) + i)

        if (i % log_every == 0) & (i > 0):
            # Display the information about average validation loss and area under ROC curve
            print('''[Epoch: {0} / {1} | Batch : {2} / {3} ]| Avg Val Loss {4} | Val AUC : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(val_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )
    # Add information to the writer about total epochs and Area under ROC curve
    writer.add_scalar('Val/AUC_epoch', auc, epoch + i)
    # Find mean area under ROC curve and validation loss
    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)

    return val_loss_epoch, val_auc_epoch

def _train_model(
    model,
    train_loader,
    epoch,
    num_epochs,
    optimizer,
    criterion,
    writer,
    current_lr,
    log_every=100,
    device='cpu',
    use_amp=False,
    scaler=None,
):
    
    # Set to train mode
    model.train()

    # Initialize the predicted probabilities
    y_probs = []
    # Initialize the groundtruth labels
    y_gt = []
    # Initialize the loss between the groundtruth label
    # and the predicted probability
    losses = []

    # Iterate over the training dataset
    for i, batch in enumerate(train_loader):
        if batch is None:
            continue
        images, label = batch
        # Reset the gradient by zeroing it
        optimizer.zero_grad()
        
        # If GPU is available, transfer the images and label
        # to the GPU
        if device != 'cpu':
            images = [image.to(device, non_blocking=True) for image in images]
            label = label.to(device, non_blocking=True)

        # Obtain the prediction using the model
        with autocast(enabled=bool(use_amp and device != 'cpu')):
            output = model(images)

        # Evaluate the loss by comparing the prediction
        # and groundtruth label
        loss = criterion(output, label)
        if scaler is not None and bool(use_amp and device != 'cpu'):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Perform a backward propagation
            loss.backward()
            # Modify the weights based on the error gradient
            optimizer.step()

        # Add current loss to the list of losses
        loss_value = loss.item()
        losses.append(loss_value)

        # Find probabilities from output using sigmoid function
        probas = torch.sigmoid(output)

        # Add current groundtruth label to the list of groundtruths
        y_gt.extend(label.detach().cpu().view(-1).numpy().astype(int).tolist())
        # Add current probabilities to the list of probabilities
        y_probs.extend(probas.detach().cpu().view(-1).numpy().tolist())

        try:
            # Try finding the area under ROC curve
            auc = metrics.roc_auc_score(y_gt, y_probs)
        except:
            # Use default value of area under ROC curve as 0.5
            auc = 0.5
        
        # Add information to the writer about training loss and Area under ROC curve
        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            # Display the information about average training loss and area under ROC curve
            print('''[Epoch: {0} / {1} | Batch : {2} / {3} ]| Avg Train Loss {4} | Train AUC : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )
    # Add information to the writer about total epochs and Area under ROC curve
    writer.add_scalar('Train/AUC_epoch', auc, epoch + i)

    # Find mean area under ROC curve and training loss
    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)

    return train_loss_epoch, train_auc_epoch

def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
