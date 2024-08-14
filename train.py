import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import VisionTransformer
from data import get_dataloaders
import matplotlib.pyplot as plt

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataloader.dataset)
        accuracy = correct / total
        return epoch_loss, accuracy
    
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = './cifar-10-batches-py'
    BATCH_SIZE = 64
    EPOCHS = 200
    LEARNING_RATE = 5e-4
    EMBED_DIM = 128
    PATCH_SIZE = 4
    NUM_PATCHES = (32 // PATCH_SIZE) ** 2
    DROPOUT = 0.4
    IN_CHANNELS = 3
    NHEAD = 8
    ACTIVATION = 'gelu'
    NUM_LAYERS = 6
    NUM_CLASSES = 10
    WEIGHT_DECAY = 5e-4
    PATIENCE = 10

    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)

    # Initialize model
    model = VisionTransformer(
        embed_dim=EMBED_DIM,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        dropout=DROPOUT,
        in_channels=IN_CHANNELS,
        nhead=NHEAD,
        activation=ACTIVATION,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Total parameters: 6588042
    # Trainable parameters: 6588042
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Initialize the learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        # Print the current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{EPOCHS}, Current Learning Rate: {current_lr:.6f}")
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best.pth')
            print(f"Saved model with validaiton loss: {val_loss:.4f} and accuracy: {val_accuracy:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        # Step the scheduler
        scheduler.step()
    
    # Test the model
    test_model(model, test_loader, device)

    # Plotting the results
    epochs_range = range(len(train_losses))

    # Plotting Training and Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # Plotting Training and Validation Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.title('Training and Validation Accuracy')
    plt.show()
