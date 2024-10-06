import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


training_data = datasets.ImageFolder(root='training_set', transform=transform)
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 clase (câine și pisică)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instanțiem modelul, funcția de pierdere și optimizatorul
model = CNN()
criterion = nn.CrossEntropyLoss()  # Pierderea pentru clasificare cu mai multe clase
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f'Epoca {epoch+1}/{num_epochs}', unit='batch')

        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix({'Pierdere': running_loss / len(progress_bar)})


        print(f"Epoca {epoch+1}, Pierdere: {running_loss / len(train_loader)}")


train_model(5)


torch.save(model.state_dict(), 'cat_dog_classifier.pth')

print("Antrenament finalizat și modelul salvat.")
