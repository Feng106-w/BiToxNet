import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import time
import copy

def load_features(trad_path, esm_path):
    trad_df = pd.read_csv(trad_path)
    trad_features = trad_df.iloc[:, :-1].values  
    labels = trad_df.iloc[:, -1].values            
    
    esm_df = pd.read_csv(esm_path, header=0)
    esm_features = esm_df.values
    
    return trad_features, esm_features, labels


train_trad, train_esm, train_labels = load_features( )

test_trad, test_esm, test_labels = load_features( )


train_trad_tensor = torch.FloatTensor(train_trad)
train_esm_tensor = torch.FloatTensor(train_esm)
train_labels_tensor = torch.LongTensor(train_labels)

test_trad_tensor = torch.FloatTensor(test_trad)
test_esm_tensor = torch.FloatTensor(test_esm)
test_labels_tensor = torch.LongTensor(test_labels)

full_train_dataset = TensorDataset(train_trad_tensor, train_esm_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_trad_tensor, test_esm_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size= , shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = " "
os.makedirs(save_dir, exist_ok=True)


def calc_loss(y_pred, labels):
    CE = nn.CrossEntropyLoss()
    ce = CE(y_pred, labels)
    return ce / y_pred.shape[0] 


all_test_results = []

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state= )

for fold, (train_idx, val_idx) in enumerate(skf.split(train_trad, train_labels)):
    start_time = time.time()
    
    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size= , shuffle=True)
    val_loader = DataLoader(val_subset, batch_size= , shuffle=False)
    
    model = dvib_ban(
        trad_input_dim,
        esm_input_dim,
        reduce_dim,
        hidden_size,
        dropout,
        ban_heads
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr= , weight_decay= )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    
    num_epochs = 100
    best_val_acc = 0.0
    best_model = None
    patience = 20
    no_improve = 0
    fold_best_epoch = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        
        for trad, esm, labels in train_loader:
            trad, esm = trad.to(device), esm.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(trad, esm)
            loss = calc_loss(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item() * trad.size(0)
        
        
        train_loss /= len(train_loader.dataset)
        train_acc = 100. * train_correct / len(train_loader.dataset)
        
        
        val_loss, val_acc, val_sen, val_spe, val_f1, val_mcc, val_auc = evaluate(model, val_loader, device)
        
        
        scheduler.step()
        
        
        print(f"\n Epoch {epoch+1}/{num_epochs} | Fold {fold+1}/{n_splits}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f} | MCC: {val_mcc:.4f} | AUC: {val_auc:.4f}")
        
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            no_improve = 0
            fold_best_epoch = epoch + 1
        else:
            no_improve += 1

        
        if no_improve >= patience:
            print(f"Early stopping triggered! Validation set has not improved after {epoch+1} epochs.")
            break
    
    model.load_state_dict(best_model)
    
    test_loss, test_acc, test_sen, test_spe, test_f1, test_mcc, test_auc = evaluate(model, test_loader, device)
    

    fold_result = {
        'fold': fold+1,
        'best_epoch': fold_best_epoch,
        'val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_sen': test_sen,   
        'test_spe': test_spe,   
        'test_f1': test_f1,
        'test_mcc': test_mcc,
        'test_auc': test_auc,
        'train_time': time.time() - start_time
    }
    
    all_test_results.append(fold_result)

    
    
    model_path = os.path.join(save_dir, f"best_model_fold{fold+1}.pth")
    torch.save({
        'fold': fold+1,
        'epoch': fold_best_epoch,
        'model_state_dict': model.state_dict(),
        'val_acc': best_val_acc
    }, model_path)
    print(f"Save the best model of fold {fold+1} to: {model_path}")
    
    print(f"The results of the {fold + 1}th test:")
    print(f"test accuracy: {test_acc:.2f}% | F1: {test_f1:.4f} | MCC: {test_mcc:.4f} | AUC: {test_auc:.4f}")
    print(f"training time: {fold_result['train_time']:.2f}second")


avg_results = {
    'fold': 'mean',
    'val_acc': np.mean([r['val_acc'] for r in all_test_results]),
    'test_loss': np.mean([r['test_loss'] for r in all_test_results]),
    'test_acc': np.mean([r['test_acc'] for r in all_test_results]),
    'test_sen': np.mean([r['test_sen'] for r in all_test_results]),  
    'test_spe': np.mean([r['test_spe'] for r in all_test_results]),  
    'test_f1': np.mean([r['test_f1'] for r in all_test_results]),
    'test_mcc': np.mean([r['test_mcc'] for r in all_test_results]),
    'test_auc': np.mean([r['test_auc'] for r in all_test_results]),
    'train_time': np.mean([r['train_time'] for r in all_test_results])
}

std_results = {
    'fold': 'std',
    'val_acc': np.std([r['val_acc'] for r in all_test_results]),
    'test_loss': np.std([r['test_loss'] for r in all_test_results]),
    'test_acc': np.std([r['test_acc'] for r in all_test_results]),
    'test_sen': np.std([r['test_sen'] for r in all_test_results]),  
    'test_spe': np.std([r['test_spe'] for r in all_test_results]),  
    'test_f1': np.std([r['test_f1'] for r in all_test_results]),
    'test_mcc': np.std([r['test_mcc'] for r in all_test_results]),
    'test_auc': np.std([r['test_auc'] for r in all_test_results]),
    'train_time': np.std([r['train_time'] for r in all_test_results])
}


print(f"Average test accuracy rate: {avg_results['test_acc']:.2f}% ± {std_results['test_acc']:.2f}")
print(f"Average sensitivity (SEN): {avg_results['test_sen']:.4f} ± {std_results['test_sen']:.4f}")
print(f"Average specificity (SPE): {avg_results['test_spe']:.4f} ± {std_results['test_spe']:.4f}")
print(f"Average test F1 score: {avg_results['test_f1']:.4f} ± {std_results['test_f1']:.4f}")
print(f"Average test MCC: {avg_results['test_mcc']:.4f} ± {std_results['test_mcc']:.4f}")
print(f"Average test AUC: {avg_results['test_auc']:.4f} ± {std_results['test_auc']:.4f}")
print(f"Average training time: {avg_results['train_time']:.2f}秒 ± {std_results['train_time']:.2f}")