import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn

# Select the initialization method
init_method = 'rsvd'

if init_method == 'random':
    from decomp.hotcake_random import HighTkd2ConvRandom as hotcake
elif init_method == 'rsvd':
    from decomp.hotcake_rsvd import HighTkd2ConvRSvd as hotcake
elif init_method == 'vbmf':
    from decomp.hotcake_vbmf import HighTkd2ConvVbmf as hotcake

# Detect devices
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)

# Load the model
model = torchvision.models.alexnet(pretrained=False)
model.load_state_dict(torch.load('alexnet_cifar10.pth'))

# Decompose the selected layers
model.features[6] = hotcake(model.features[6], k11=16, k12=12, r31=3, r32=3, r4=20)
model.features[8] = hotcake(model.features[8], k11=16, k12=24, r31=3, r32=3, r4=30)
model.features[10] = hotcake(model.features[10], k11=16, k12=16, r31=3, r32=3, r4=30)

# Load model to the device and print the model
model.to(device)
print(model)

# Load the dataset
dataset = torchvision.datasets.CIFAR10(root='~/Data', train=True, transform=T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]), download=True)
test_dataset = torchvision.datasets.CIFAR10(
    root='~/Data', train=False, transform=T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]), download=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# Define the optimizer and the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()


# Training the model after decomposition
def fine_tune(epoch, log_interval=200):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{:0>5d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx * len(data) / len(data_loader.dataset), loss.detach().item()))

torch.save(model.state_dict(), './alexnet_hotcake.pth')


# Test the compressed model after fine-tuning
@torch.no_grad()
def test():
    model.eval()
    val_loss, correct = 0, 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum()

    val_loss /= len(test_loader)

    accuracy = 100. * correct.to(torch.float32) / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(test_loader.dataset), accuracy))


# Set training Epochs
epochs = 30
for epoch in range(epochs):
    fine_tune(epoch, 50)
    test()
