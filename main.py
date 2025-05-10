
#runs the training and testing of the model

from code.model import simpleOverallModel
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from code.load_data import load_haxby_data

def training_cnn(model,training_data, num_epochs=10, batch_size=32, learning_rate=1e-3):
    print("Training CNN model...")
    
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output, embedding, logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
    # Evaluate on training data
    model.eval()
    correct = 0
    total = 0
    cumulative_loss = 0.0

    with torch.no_grad():
        for inputs, labels in training_data:
            inputs, labels = inputs.to(device), labels.to(device)
            _, _, logits = model(inputs)
            loss = criterion(logits, labels)
            cumulative_loss += loss.item() * inputs.size(0)

            # if labels are one-hot, convert to class indices
            if labels.dim() > 1:
                labels_idx = labels.argmax(dim=1)
            else:
                labels_idx = labels

            preds = logits.argmax(dim=1)
            correct += (preds == labels_idx).sum().item()
            total += labels.size(0)

    final_loss = cumulative_loss / total
    accuracy = correct / total

    print(f"Final Training Loss: {final_loss:.4f}")
    print(f"Training Accuracy: {accuracy:.4f}")

    return final_loss, accuracy

def eval_cnn(model, test_data):
    print("Evaluating CNN model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_data:
            inputs, labels = inputs.to(device), labels.to(device)
            _, _, logits = model(inputs)

            # if labels are one-hot, convert to class indices
            if labels.dim() > 1:
                labels_idx = labels.argmax(dim=1)
            else:
                labels_idx = labels

            preds = logits.argmax(dim=1)
            correct += (preds == labels_idx).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy

if __name__ == "__main__":
    
    print("Beginning training and testing of the model")
    
    training_datasets = load_haxby_data()
    sample_dataset = training_datasets[0]
    
    
    
    model = simpleOverallModel(num_classes=9, input_dim=(3, 64, 64))
    cnn = model.cnn
    gan = model.gan
    
    #Training and creating data for CNN
    # Convert the sample dataset to tensors and wrap in a DataLoader
    train_data, train_labels = sample_dataset["X"], sample_dataset["y"]
    # Extract only the string label from each tuple
    labels_str = [lbl for lbl, _ in train_labels]

    # Build a mapping from each unique label to an index
    unique_labels = sorted(set(labels_str))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    # One‐hot encode into a list of vectors
    one_hot_labels = []
    for lbl in labels_str:
        vec = [0] * len(unique_labels)
        vec[label_to_idx[lbl]] = 1
        one_hot_labels.append(vec)

    # Replace train_labels with the one‐hot encoded vectors
    train_labels = one_hot_labels
    
    
    import numpy as np
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    # Convert list of arrays to numpy arrays for better performance
    # Convert to numpy arrays
    data_arr = np.array(train_data)
    labels_arr = np.array(train_labels)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data_arr, labels_arr, test_size=0.2, shuffle=True, random_state=42
    )

    # Convert to tensors
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_labels_tensor = torch.tensor(y_train, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_labels_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create datasets
    train_dataset = TensorDataset(train_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_tensor, test_labels_tensor)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    training_cnn(cnn, train_loader, num_epochs=10, batch_size=32, learning_rate=1e-4)
    eval_cnn(cnn, test_loader)
    
    #Training and creating data for GAN
    
    