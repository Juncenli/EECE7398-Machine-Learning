from PIL import Image
from pickle import FALSE
from utils import *
from Model import *
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import sys

def preprocessing(train_test):
    '''
    description: preprocessing the cifar-10 dataset
    param {train_test:decide to return trainloader or testloader}
    return trainloader,testloader
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ) 
    
    if train_test == "train":
        training_data = torchvision.datasets.CIFAR10( 
            root='./data', 
            train=True,
            download=FALSE, 
            transform=transform
        )

        '''
        We pass the Dataset as an argument to DataLoader. 
        This wraps an iterable over our dataset, and supports automatic batching, sampling, shuffling and multiprocess data loading. 
        Here we define a batch size of 64, i.e. each element in the dataloader iterable will return a batch of 64 features and labels.
        '''

        loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )

    if train_test == "test":
        test_data = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False,
            download=FALSE, 
            transform=transform
        )
        loader = torch.utils.data.DataLoader(
            test_data, 
            batch_size=BATCH_SIZE,
            shuffle=False, 
            num_workers=0
        )

    return loader

    '''
    description: training deep model. Saving in path stored in utils.py
    param {
        EPOCH,
        trainloader:to provide training data. A pytorch dataloader
        testloader:to provide test data. A pytorch dataloader
        LR:learning rate
    }
    return {*}
    '''

def train(EPOCH, trainloader, testloader, LR):
    # To train a model, we need a loss function and an optimizer.
    model = Net().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = LR)

    #format accuracy output
    print_format="{0:<6}{1:<12.4f}{2:<12.4f}{3:<11.4f}{4:<10.4f}" 
    
    '''
    In a single training loop, the model makes predictions on the training dataset (fed to it in batches), 
    and backpropagates the prediction error to adjust the model’s parameters.
    '''

    for epoch in range(EPOCH):  

        running_loss = 0.0
        best_loss=10000
        best_acc=0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #train statisctics
            _, train_predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (train_predicted == labels).sum().item()
            
            # print statistics
                        
            running_loss += loss.item()
            if i==0:
                print('Loop ','Train Loss ','Train Acc% ','Test Loss ','Test Acc%')
            if i % 200 == 199:    # print every 200 batches
                #test statisctics
                test_correct = 0
                test_total = 0
                test_running_loss=0
                test_running_count=0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        test_outputs = model(images)
                        test_loss=loss_fn(test_outputs,labels)
                        test_running_loss+=test_loss.item()
                        test_running_count+=1
                        _, predicted = torch.max(test_outputs.data, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()
                        
                
                acc=100 * test_correct / test_total
                train_acc=100 * correct / total
                print(print_format.format(str(epoch + 1), running_loss / 200,train_acc,test_running_loss/test_running_count,acc))
                
                if running_loss<best_loss and acc>best_acc:
                    best_acc=acc
                    torch.save(model.state_dict(), PATH)
                    
                running_loss = 0.0
                test_running_loss=0.0
                test_running_count=0
    print('train finished')

def test(img_path):
    '''
    description: to test single picture. Print the result in consule.
    param {
        img_path: image file path
    }
    return {*}
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #C*H*W
    img=Image.open(img_path).convert('RGB')
    img=transform(img).unsqueeze(0) #(1,3,32,32)
    #print(img.shape)
    model = Net().to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    output=model(img)
    _, predicted = torch.max(output, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted]))

def main():
    '''
    description: main function. Program starts here.
    param {*}
    return {*}
    '''
    if sys.argv[1] == 'train':
        trainloader = preprocessing("train")
        testloader = preprocessing("test")
        train(EPOCH = EPOCH, trainloader = trainloader, testloader = testloader, LR = 0.001)
    elif sys.argv[1] == 'test' or 'predict':
        test(sys.argv[2])
    else:
        print('wrong parameter')

if __name__ == '__main__':
    main()