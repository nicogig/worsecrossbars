"""
mlp_trainer:
A backend module used to instantiate the MNIST dataset and train a PyTorch model on it.
"""

from pathlib import Path
import numpy as np
import torch
from torch import device
from torch import cuda
from torch import Generator
from torch import manual_seed
from torch.optim import RMSprop, Adam
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from worsecrossbars.backend.mlp_generator_pytorch import MNIST_MLP


def get_data_loaders(**kwargs):
    """
    -
    """

    # Unpacking keyword arguments
    batch_size = kwargs.get("batch_size", 100)
    seed = kwargs.get("seed", 42)
    validation_size = kwargs.get("validation_size", 0.25)
    data_directory = kwargs.get("data_directory",
                                str(Path.home().joinpath("worsecrossbars", "utils")))
    shuffle = kwargs.get("shuffle", True)
    num_workers = kwargs.get("num_workers", 1)
    pin_memory = kwargs.get("pin_memory", True)

    # Validating arguments
    if isinstance(validation_size, int):
        validation_size = float(validation_size)
    if not isinstance(validation_size, float) or validation_size < 0 or validation_size > 1:
        raise ValueError("\"validation_size\" argument should be a real number comprised between " +
                         "0 and 1.")

    # Defining transforms, including standard MNIST normalisation
    normalize = Normalize((0.1307,), (0.3081,))
    transform = Compose([ToTensor(),normalize])

    # Loading the datasets
    full_training_dataset = datasets.MNIST(root=data_directory, train=True, download=True,
                                           transform=transform)
    test_dataset = datasets.MNIST(root=data_directory, train=False, download=True,
                                  transform=transform)

    # Splitting the dataset 
    size_full_dataset = len(full_training_dataset)
    size_validation = int(np.floor(validation_size * size_full_dataset))
    size_training = size_full_dataset - size_validation
    training_dataset, validation_dataset = random_split(full_training_dataset,
                                                        [size_training, size_validation],
                                                        generator=Generator().manual_seed(seed))

    training_loader = DataLoader(training_dataset, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=num_workers,
                                 pin_memory=pin_memory)

    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers,
                                   pin_memory=pin_memory)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory)

    return training_loader, validation_loader, test_loader


# def training(epoch):

#     model.train()

#     for data, label in training_loader:

#         if cuda.is_available():
#             data, label = data.cuda(), label.cuda()

#         optimiser.zero_grad()

#         output = model(data)
#         loss = cross_entropy(output, label)

#         loss.backward()
#         optimiser.step()

#         training_loss += loss.item()

#         if batch_index % log_interval == 0:

#             print("Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                 epoch, batch_index * len(data), len(training_loader.dataset),
#                 100. * batch_index / len(training_loader), loss.item()))
            
#             training_losses.append(loss.item())
#             training_counter.append((batch_index*100) + ((epoch-1)*len(training_loader.dataset)))

#             torch.save(model.state_dict(), 'results/model.pth')
#             torch.save(optimiser.state_dict(), 'results/optimizer.pth')


# def training(epoch):

#     model.train()

#     for batch_index, (data, label) in enumerate(training_loader):

#         if cuda.is_available():
#             data, label = data.cuda(), label.cuda()

#         optimiser.zero_grad()

#         output = model(data)
#         loss = cross_entropy(output, label)

#         loss.backward()
#         optimiser.step()

#         if batch_index % log_interval == 0:

#             print("Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                 epoch, batch_index * len(data), len(training_loader.dataset),
#                 100. * batch_index / len(training_loader), loss.item()))
            
#             training_losses.append(loss.item())
#             training_counter.append((batch_index*100) + ((epoch-1)*len(training_loader.dataset)))

#             torch.save(model.state_dict(), 'results/model.pth')
#             torch.save(optimiser.state_dict(), 'results/optimizer.pth')


# def validation(epoch):

#     model.eval()

#     for batch_index, (data, label) in enumerate(validation_loader):
#         f



# Training with Validation
# epochs = 5
# min_valid_loss = np.inf
 
# for e in range(epochs):
#     train_loss = 0.0
#     for data, labels in trainloader:
#         # Transfer Data to GPU if available
#         if torch.cuda.is_available():
#             data, labels = data.cuda(), labels.cuda()
         
#         # Clear the gradients
#         optimizer.zero_grad()
#         # Forward Pass
#         target = model(data)
#         # Find the Loss
#         loss = criterion(target,labels)
#         # Calculate gradients
#         loss.backward()
#         # Update Weights
#         optimizer.step()
#         # Calculate Loss
#         train_loss += loss.item()
     
#     valid_loss = 0.0
#     model.eval()     # Optional when not using Model Specific layer
#     for data, labels in validloader:
#         # Transfer Data to GPU if available
#         if torch.cuda.is_available():
#             data, labels = data.cuda(), labels.cuda()
         
#         # Forward Pass
#         target = model(data)
#         # Find the Loss
#         loss = criterion(target,labels)
#         # Calculate Loss
#         valid_loss += loss.item()
 
#     print(f'Epoch {e+1} \t\t Training Loss: {\
#     train_loss / len(trainloader)} \t\t Validation Loss: {\
#     valid_loss / len(validloader)}')
     
#     if min_valid_loss > valid_loss:
#         print(f'Validation Loss Decreased({min_valid_loss:.6f\
#         }--->{valid_loss:.6f}) \t Saving The Model')
#         min_valid_loss = valid_loss
         
#         # Saving State Dict
#         torch.save(model.state_dict(), 'saved_model.pth')


def test():

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():

        for data, label in test_loader:

            output = model(data)
            test_loss += cross_entropy(output, label, size_average=False).item()
            prediction = output.data.max(1, keepdim=True)[1]
            correct += prediction.eq(label.data.view_as(prediction)).sum()
        
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print("\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct, 
          len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":

    random_seed = 42
    number_hidden_layers = 2
    number_epochs = 10

    # Device configuration
    device = device("cuda" if cuda.is_available() else "cpu")

    # If reproducibility is desired, a seed must be set, and cuDNN must be disabled, as it uses
    # nondeterministic algorithms
    torch.backends.cudnn.enabled = False
    # manual_seed(random_seed)

    # Initialising MNIST_MLP
    model = MNIST_MLP(number_hidden_layers)

    # Sending network model to GPU if available
    if cuda.is_available():
        model = model.cuda()
    
    # optimiser = RMSprop(network.parameters(), lr=learning_rate, momentum=momentum)
    # optimiser = RMSprop(network.parameters())
    optimiser = Adam(model.parameters())

    training_loader, validation_loader, test_loader = get_data_loaders()

    training_losses = []
    training_counter = []
    validation_losses = []
    validation_counter = []
    test_losses = []
    test_counter = [i*len(training_loader.dataset) for i in range(number_epochs + 1)]

    # for param in model.parameters():
    #     print(param.shape)
    weights = []
    for param in model.parameters():
        weights.append(param)
        #print(param.shape)
    print(weights[2])

    for epoch in range(1, number_epochs + 1):

        # Training step
        training_loss = 0.0
        model.train()

        for data, label in training_loader:

            if cuda.is_available():
                data, label = data.cuda(), label.cuda()

            optimiser.zero_grad()

            output = model(data)
            loss = cross_entropy(output, label)

            loss.backward()
            optimiser.step()

            training_loss += loss.item()

        # Validation step
        validation_loss = 0.0
        model.eval()

        for data, label in validation_loader:

            if cuda.is_available():
                data, label = data.cuda(), label.cuda()
            
            output = model(data)
            loss = cross_entropy(output, label)

            validation_loss += loss.item()
        
        # print(f"Epoch {epoch} \t\t Training Loss: {training_loss / len(training_loader)}" +
        #       f"\t\t Validation Loss: {validation_loss / len(validation_loader)}")
        
        test()

    weights = []
    for param in model.parameters():
        weights.append(param)
        #print(param.shape)
    print(weights[2])






# def train_mlp(dataset, model, epochs, batch_size):

#     if not isinstance(epochs, int) or epochs < 1:
#         raise ValueError("\"epochs\" argument should be an integer greater than 1.")

#     if not isinstance(batch_size, int) or batch_size < 1:
#         raise ValueError("\"batch_size\" argument should be an integer greater than 1.")

#     model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

#     # Training with validation test
#     mlp_history = model.fit(dataset[0][2], dataset[0][3], epochs=epochs, batch_size=batch_size,
#                             validation_data=(dataset[0][0], dataset[0][1]), verbose=0)
#     mlp_test_loss, mlp_test_acc = model.evaluate(dataset[1][0], dataset[1][1], verbose=0)

#     # Extracting network weights
#     mlp_weights = model.get_weights()

#     return mlp_weights, mlp_history, mlp_test_loss, mlp_test_acc
