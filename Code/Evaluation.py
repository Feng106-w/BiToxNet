import torch
import numpy as np
from sklearn import metrics

def evaluate(model, data_loader, device, threshold):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for trad, esm, labels in data_loader:
            trad, esm = trad.to(device), esm.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(trad, esm)
            loss = calc_loss(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1] > threshold).long()
            
            correct += (preds == labels).sum().item()
            total_loss += loss.item() * trad.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader.dataset)
    acc = 100. * correct / len(data_loader.dataset)
    
    tn, fp, fn, tp = metrics.confusion_matrix(all_labels, all_preds).ravel()
    sen = tp / (tp + fn)  
    spe = tn / (tn + fp)  
    
    f1 = metrics.f1_score(all_labels, all_preds)
    mcc = metrics.matthews_corrcoef(all_labels, all_preds)
    auc = metrics.roc_auc_score(all_labels, all_preds)
    
    return avg_loss, acc, sen, spe, f1, mcc, auc, all_preds, all_labels