import time

import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_gsm_structured(net, epochs, batch_size, lr, reg, nonzero_ratio, log_every_n=50, net_name = 'struct_gsm.pt'):
    """
    Training a network
    :param net: Network for training
    :param epochs: Number of epochs in total.
    :param batch_size: Batch size for training.
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.99, weight_decay=reg, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)

    global_steps = 0
    start = time.time()

    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            '''Struct GSM'''
            to_concat = []
            for n, m in net.named_modules():
                if isinstance(m, nn.Conv2d):
                    v = m.weight.data
                    g = m.weight.grad.data
                    to_concat.append(torch.mean(torch.abs(g), dim =(1, 2, 3))*
                                     torch.mean(torch.abs(v), dim =(1, 2, 3)).view(-1))
                if isinstance(m, nn.Linear):
                    v = m.weight.data
                    g = m.weight.grad.data
                    to_concat.append(torch.mean(torch.abs(g), dim =(0))*
                                     torch.mean(torch.abs(v), dim =(0)).view(-1))            

            all_para = torch.cat(to_concat)
            num_params = all_para.size(0)
            nz = int(nonzero_ratio * num_params)
            top_values, _ = torch.topk(all_para, nz)
            thresh = top_values[-1]
            for n, m in net.named_modules():
                
                if isinstance(m, nn.Conv2d): 
                    v = m.weight.data
                    g = m.weight.grad.data
                    mask = torch.mean(torch.abs(v * g), dim =(1, 2, 3)) >= thresh
                    mask = torch.reshape(mask, (mask.size(0), 1, 1, 1))
                    m.weight.data = m.weight.data*mask
                if isinstance(m, nn.Linear):
                    v = m.weight.data
                    g = m.weight.grad.data
                    mask = torch.mean(torch.abs(v * g), dim =(0)) >= thresh
                    mask = torch.reshape(mask, (1, mask.size(0)))
                    m.weight.data = m.weight.data*mask
                    
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % log_every_n == 0:
                end = time.time()
                num_examples_per_second = log_every_n * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()

        scheduler.step()

        """
        Start the testing code.
        """
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving...")
            torch.save(net.state_dict(), 'saved_models/'+net_name)

    
def train_gsm_unstructured(net, epochs, batch_size, lr, reg, nonzero_ratio, log_every_n=50, net_name = 'unstruct_gsm.pt'):
    """
    Training a network
    :param net: Network for training
    :param epochs: Number of epochs in total.
    :param batch_size: Batch size for training.
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.99, weight_decay=reg, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)

    global_steps = 0
    start = time.time()

    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            to_concat_g = []
            to_concat_v = []
            for n, m in net.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    v = m.weight.data
                    g = m.weight.grad.data
                    to_concat_g.append(g.view(-1))
                    to_concat_v.append(v.view(-1))
            all_g = torch.cat(to_concat_g)
            all_v = torch.cat(to_concat_v)
            metric = torch.abs(all_g * all_v)
            num_params = all_v.size(0)
            nz = int(nonzero_ratio * num_params)
            top_values, _ = torch.topk(metric, nz)
            thresh = top_values[-1]
            '''GSM'''
            for n, m in net.named_modules():
                if isinstance(m, nn.Conv2d):
                    v = m.weight.data
                    g = m.weight.grad.data
                    mask = (torch.abs(v * g) >= thresh).type(torch.cuda.FloatTensor)
                    m.weight.data = m.weight.data*mask

            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % log_every_n == 0:
                end = time.time()
                num_examples_per_second = log_every_n * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()

        scheduler.step()

        """
        Start the testing code.
        """
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving...")
            torch.save(net.state_dict(), 'saved_models/'+net_name)
    

def train(net, epochs, batch_size, lr, reg, log_every_n=50, net_name = 'base.pt'):
    """
    Training a network
    :param net: Network for training
    :param epochs: Number of epochs in total.
    :param batch_size: Batch size for training.
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.875, weight_decay=reg, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)

    global_steps = 0
    start = time.time()

    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            global_steps += 1

            if global_steps % log_every_n == 0:
                end = time.time()
                num_examples_per_second = log_every_n * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()

        scheduler.step()

        """
        Start the testing code.
        """
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(testloader)
        val_acc = correct / total
        print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving...")
            torch.save(net.state_dict(), 'saved_models/'+net_name)
            
    
def test(net):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    print("Test Loss=%.4f, Test accuracy=%.4f" % (test_loss / (num_val_steps), val_acc))



