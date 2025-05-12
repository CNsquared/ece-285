
#runs the training and testing of the model

from code.model import simpleOverallModel
import torch
from code.load_data import load_haxby_data, create_cnn_datasets
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


def training_cnn(model, train_loader, num_epochs=10, batch_size=32, learning_rate=1e-3):
    print("Training CNN model...")
    
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    num_classes = model.num_classes


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        
         
        model.train()
        running_total_loss = 0.0
        running_loss = torch.zeros(num_classes)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for inputs, labels in loop:
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output, embedding, logits = model(inputs)
            
            # If labels are one‐hot, convert to class indices
            labels_idx = labels.argmax(dim=1)

            # 1) Inverse frequency weights
            class_counts = torch.bincount(labels_idx, minlength=num_classes).float()
            freq_weights = 1.0 / (class_counts + 1e-6)
            freq_weights = freq_weights / freq_weights.sum() * num_classes

            # 2) Update running loss per class
            with torch.no_grad():
                loss_per_example = F.cross_entropy(logits, labels_idx, weight=freq_weights, reduction='none')
                for cls in range(num_classes):
                    mask = (labels_idx == cls)
                    if mask.any():
                        running_loss[cls] = 0.9 * running_loss[cls] + 0.1 * loss_per_example[mask].mean()

            # 3) Compute final weights: combining frequency and difficulty
            difficulty_weights = running_loss / running_loss.sum()
            combined_weights = freq_weights * difficulty_weights
            combined_weights = combined_weights / combined_weights.sum() * num_classes

            # 4) Use updated weights in the loss
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1,weight=combined_weights)
            
            
            loss = criterion(logits, labels_idx)
            loss.backward()
            optimizer.step()

            running_total_loss += loss.item() * inputs.size(0)
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
    


def eval_cnn(model, test_data):

    print("Evaluating CNN model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # containers
    y_true = []
    y_pred = []
    misclass_counts = defaultdict(lambda: defaultdict(int))

    # evaluation loop
    with torch.no_grad():
        for inputs, labels in test_data:
            inputs, labels = inputs.to(device), labels.to(device)
            output, _, logits = model(inputs)
            # convert one‐hot if needed
            if labels.dim() > 1:
                labels_idx = labels.argmax(dim=1)
            else:
                labels_idx = labels
            preds = output.argmax(dim=1)

            for p, t in zip(preds, labels_idx):
                ti = t.item(); pi = p.item()
                y_true.append(ti)
                y_pred.append(pi)
                if pi != ti:
                    misclass_counts[ti][pi] += 1

    # classes and counts
    classes = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # compute per‐class accuracy
    class_total = cm.sum(axis=1)
    class_correct = np.diag(cm)
    per_class_acc = {c: class_correct[i]/class_total[i] if class_total[i] > 0 else 0.0
                     for i, c in enumerate(classes)}

    # overall, balanced, weighted
    total_samples = sum(class_total)
    overall_acc = class_correct.sum() / total_samples
    balanced_acc = np.mean(list(per_class_acc.values()))
    weighted_acc = sum(per_class_acc[c] * class_total[i]
                       for i, c in enumerate(classes)) / total_samples

    # print table header
    print("\nPer-class results:")
    header = f"{'Class':>5} | {'Total':>5} | {'Correct':>7} | {'Accuracy':>8} | {'Most Confused':>14}"
    print(header)
    print("-" * len(header))
    for i, c in enumerate(classes):
        total = class_total[i]
        correct = class_correct[i]
        acc = per_class_acc[c]
        # find most confused
        if misclass_counts[c]:
            mc, cnt = max(misclass_counts[c].items(), key=lambda x: x[1])
            mc_str = f"{mc} ({cnt})"
        else:
            mc_str = "—"
        print(f"{c:>5} | {total:>5} | {correct:>7} | {acc:>8.4f} | {mc_str:>14}")

    # print confusion matrix
    print("\nConfusion Matrix:")
    # header row
    hdr = "      " + "".join(f"{c:>5}" for c in classes)
    print(hdr)
    for i, c in enumerate(classes):
        row = "".join(f"{cm[i,j]:>5}" for j in range(len(classes)))
        print(f"{c:>5} |{row}")

    # print summary metrics
    print("\nSummary accuracy metrics:")
    print(f"  Overall accuracy:           {overall_acc:.4f}")
    print(f"  Balanced accuracy (macro):  {balanced_acc:.4f}")
    print(f"  Weighted accuracy:          {weighted_acc:.4f}")

    return {
        'overall_acc': overall_acc,
        'balanced_acc': balanced_acc,
        'weighted_acc': weighted_acc,
        'per_class_acc': per_class_acc,
        'confusion_matrix': cm,
        'misclass_counts': misclass_counts
    }

if __name__ == "__main__":
    num_slices = 3 # number of slices per axis 

    
    print("Loading data...")
    training_datasets = load_haxby_data(n_slices=num_slices, pickle_file=False)
    
    t = training_datasets[0]
    train_data, train_labels = t["X"], t["y"]
    print(f"shape of training data: {train_data[0].shape}")
    #print(f"shape of training labels: {train_labels[0].shape}")
       
    print("Beginning training and testing of the model")
    model = simpleOverallModel(num_classes=9, input_dim=(num_slices * 3, 64, 64))
    cnn = model.cnn
    gan = model.gan
    
    # collect all sub‐datasets
    train_ds_list = []
    test_ds_list  = []
    for sample_dataset in training_datasets:
        tr_loader, te_loader = create_cnn_datasets(sample_dataset)
        train_ds_list.append(tr_loader.dataset)
        test_ds_list.append(te_loader.dataset)

    # concatenate into one big dataset
    combined_train_ds = ConcatDataset(train_ds_list)
    combined_test_ds  = ConcatDataset(test_ds_list)

    # rebuild loaders over the combined datasets
    train_loader = DataLoader(
        combined_train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        combined_test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # inspect a single batch to print the input shapes
    batch = next(iter(train_loader))
    inputs, labels = batch
    print(f"Sample input tensor shape: {inputs.shape}")
    print(f"Sample label tensor shape: {labels.shape}")
    
    training_cnn(cnn, train_loader, num_epochs=10, batch_size=32, learning_rate=1e-3)
    eval_cnn(cnn, test_loader)
        
        
        
    #Training and creating data for GAN
    
    