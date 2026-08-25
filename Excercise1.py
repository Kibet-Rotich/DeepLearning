import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class sytheticDataset(Dataset):
    def __init__(self, num_samples=1200, n_features=8):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, n_features)  # Random data with 8 features
        self.labels = torch.randint(0, 2, (num_samples,))  # Random binary labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

model = MLP(input_size=8, hidden_size=32, output_size=2).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train_model(model, dataloader, loss_fn, optimizer, num_epochs=10):
    model.train()
    total_loss = 0
    accuracy = 0

    for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            accuracy += (outputs.argmax(dim=1) == labels).sum().item()

    return total_loss / len(dataloader), accuracy / len(dataloader.dataset)


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss_fn(outputs, labels).item()

    accuracy = 100 * correct / total
    return total_loss / len(dataloader), accuracy


# Create dataset and splits ONCE outside the loop
full_dataset = sytheticDataset(num_samples=1200, n_features=8)
train_dataset, val_dataset = random_split(full_dataset, [1000, 200])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

epochs = 10

# Now the model iteratively learns the fixed dataset over time
for epoch in range(epochs):
    train_loss, train_accuracy = train_model(model, train_dataloader, loss_fn, optimizer)
    val_loss, val_accuracy = evaluate_model(model, val_dataloader)

    print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy*100:.1f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.1f}%")