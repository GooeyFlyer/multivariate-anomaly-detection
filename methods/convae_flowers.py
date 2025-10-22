# code from https://www.geeksforgeeks.org/machine-learning/implement-convolutional-autoencoder-in-pytorch-with-cuda/

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # down-samples and learns spatial features
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # up-samples (reconstructs) to original data
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid()  # ensures output values are between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def convae():

    # images are resized and converted to tensors
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.Flowers102(
        root='flowers', split='train', transform=transform, download=True)
    test_dataset = datasets.Flowers102(
        root='flowers', split='test', transform=transform)

    # batches data and shuffles before training
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)

    # uses GPU acceleration if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device being used: {device}")

    # Model and optimizer setup
    # MSELoss computes reconstruction error
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # for each batch, moves images to device, computes forward pass and loss, updates weights
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # save the model
    torch.save(model.state_dict(), 'conv_autoencoder.pth')

    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon = model(data)
            break

    # visualise
    plt.figure(dpi=250)
    fig, ax = plt.subplots(2, 7, figsize=(15, 4))
    for i in range(7):
        ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
        ax[0, i].axis('off')
        ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
        ax[1, i].axis('off')
    ax[0, 0].set_title('Original')
    ax[1, 0].set_title('Reconstructed')
    plt.show()


if __name__ == "__main__":
    convae()
