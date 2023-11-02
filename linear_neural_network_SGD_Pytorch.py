
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np



#torchvision.datasets.FashionMNIST()


def get_data_loader(training=True):
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.FashionMNIST('./data', train=True,
                                      download=True, transform=custom_transform)

    test_set = datasets.FashionMNIST('./data', train=False,
                                     transform=custom_transform)

    if training==True:
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle = False)
    return loader





def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
        )
    return model


def train_model(model, train_loader, criterion= torch.nn.CrossEntropyLoss(), T=5):
    criteria= criterion

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        total = 0
        for data in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs) #forward
            loss = criteria(outputs, labels)
            loss.backward() #backward
            optimizer.step()  #update weight

            #find accuracy trial
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # statistics for each epoch
        running_loss += loss.item()


        print(f'Train Epoch: {epoch} Accuracy: {correct}/{total} ({100 * correct // total:.2f}%)  Loss: {running_loss:.3f}')
        running_loss = 0.0


    #print('Finished Training')

    """

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """




def evaluate_model(model, test_loader, criterion= torch.nn.CrossEntropyLoss(), show_loss = True):
    criteria = criterion
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # find accuracy
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # statistics
            loss = criteria(outputs, labels)
            running_loss = loss.item()


    if show_loss == True:
        print(f'Average Loss: {running_loss:.4f} \nAccuracy: {100 * correct // total:.2f}%')
    else:
        print(f'Accuracy: {100 * correct // total:.2f} %)')





'''
for test_images, test_labels in test_loader:
    sample_image = test_images[0]    # Reshape them according to your needs.
    sample_label = test_labels[0]
output = model(sample_image)
raw_prob, choice = torch.topk(output, 3, largest=True, sorted=True)
prob = F.softmax(raw_prob, dim=1)

choice_np = choice.numpy()
prob_np = prob.detach().numpy()

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
choice_1= class_names[choice_np[0][0]]
choice_2= class_names[choice_np[0][1]]
choice_3= class_names[choice_np[0][2]]

print(f'{choice_1}: {100*prob_np[0][0]:.2f}% \n{choice_2}: {100*prob_np[0][1]:.2f}% \n {choice_3}: {100*prob_np[0][2]:.2f}%')
'''

def predict_label(model, test_images, index):
    input_is= test_images[index]
    output = model(input_is)
    raw_prob, choice = torch.topk(output, 3, largest=True, sorted=True)
    prob = F.softmax(raw_prob, dim=1)

    choice_np = choice.numpy()
    prob_np = prob.detach().numpy()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle Boot']
    choice_1 = class_names[choice_np[0][0]]
    choice_2 = class_names[choice_np[0][1]]
    choice_3 = class_names[choice_np[0][2]]

    print(
        f'{choice_1}: {100 * prob_np[0][0]:.2f}% \n{choice_2}: {100 * prob_np[0][1]:.2f}% \n {choice_3}: {100 * prob_np[0][2]:.2f}%')

if __name__ == '__main__':
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)

    model = build_model()
    print(model)

    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, criterion, 5)

    evaluate_model(model, test_loader, criterion, show_loss=True)
