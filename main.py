
#runs the training and testing of the model

from code.model import simpleOverallModel
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def training_cnn(model, num_epochs=10, batch_size=32):
    print("Training CNN model...")
    
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    
    print("Beginning training and testing of the model")
    
    model = simpleOverallModel(num_classes=10, input_dim=(1, 32, 32))
    cnn = model.cnn
    gan = model.gan
    
    
    
    
    
    
    # Import the model
    