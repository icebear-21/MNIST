import torch
from model import ICE_NET
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.notebook import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,), inplace = True)
])

train_data = datasets.FashionMNIST(root = "./data", train = True, download = True, transform = transform)
test_data = datasets.FashionMNIST(root = "./data", train = False, download = True, transform = transform)

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True, num_workers = 4)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = False, num_workers = 4)


model = ICE_NET(num_classes = 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr = 1e-4)

epochs = 10

def train(model, train_loader, epochs):
    model.train()

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {running_loss/len(train_loader):.4f}")
        
        test(model, test_loader, epoch)

def test(model, test_loader, epoch):
    model.eval()
    running_loss = 0.0
    with torch.inference_mode():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            running_loss += loss.item()

    print(f"Epoch {epoch} - Test Loss: {running_loss/len(test_loader):.4f}")

train(model, train_loader, epochs)