import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from typing import List
from nas_framework.ip_layer import decode_layer, Layer, LayerType, MAX_LENGTH


class CNNBuilder:
    def __init__(self, layers: List[Layer], num_classes: int = 10):
        self.layers = layers
        self.num_classes = num_classes

    def build(self) -> nn.Module:
        """Build PyTorch model from layers."""
        layers = []
        in_channels = 3  # CIFAR-10

        for layer in self.layers:
            if layer.layer_type == LayerType.DISABLED:
                continue
            elif layer.layer_type == LayerType.CONV:
                out_channels = layer.params['num_feature_maps']
                kernel_size = layer.params['filter_size']
                stride = layer.params['stride']
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2))
                layers.append(nn.ReLU())
                in_channels = out_channels
            elif layer.layer_type == LayerType.POOLING:
                kernel_size = layer.params['kernel_size']
                stride = layer.params['stride']
                if layer.params['pool_type'] == 'max':
                    layers.append(nn.MaxPool2d(kernel_size, stride))
                else:
                    layers.append(nn.AvgPool2d(kernel_size, stride))
            elif layer.layer_type == LayerType.FC:
                # FC layers are added at the end, but for now, collect them
                pass

        # Flatten
        layers.append(nn.Flatten())
        features = nn.Sequential(*layers)

        try:
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 32, 32)
                dummy_output = features(dummy_input)
                prev_size = dummy_output.view(1, -1).shape[1]
        except Exception:
            # If architecture goes below 1x1 due to pooling
            prev_size = 1

        fc_layers = [l for l in self.layers if l.layer_type == LayerType.FC and l.layer_type != LayerType.DISABLED]
        if not fc_layers:
            fc_layers = [Layer(LayerType.FC, num_neurons=128)]

        all_layers = list(features)

        for fc in fc_layers[:-1]:
            all_layers.append(nn.Linear(prev_size, fc.params['num_neurons']))
            all_layers.append(nn.ReLU())
            prev_size = fc.params['num_neurons']

        all_layers.append(nn.Linear(prev_size, self.num_classes))

        return nn.Sequential(*all_layers)


class IPPSOEvaluator:
    def __init__(self, num_classes: int = 10, batch_size: int = 64, epochs: int = 10, mock: bool = True):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.mock = mock
        if not mock:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            train_size = int(0.8 * len(dataset))
            fitness_size = len(dataset) - train_size
            self.train_dataset, self.fitness_dataset = random_split(dataset, [train_size, fitness_size])
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            self.fitness_loader = DataLoader(self.fitness_dataset, batch_size=200, shuffle=False)

    def evaluate(self, position) -> float:
        if self.mock:
            # Mock fitness: sum of bytes / 255 / len, as a simple proxy
            return sum(position) / (255 * len(position))
        else:
            # Actual training
            layers = []
            for i in range(0, len(position), 2):
                byte0, byte1 = position[i], position[i+1]
                layer = decode_layer(byte0, byte1)
                layers.append(layer)
            layers = [l for l in layers if l.layer_type != LayerType.DISABLED]
            if not layers or layers[-1].layer_type != LayerType.FC:
                layers.append(Layer(LayerType.FC, num_neurons=self.num_classes))
            builder = CNNBuilder(layers, self.num_classes)
            model = builder.build().to(self.device)
            for m in model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            optimizer = optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            model.train()
            for epoch in range(self.epochs):
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in self.fitness_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            return correct / total

    def retrain_and_evaluate_testset(self, position, epochs=10) -> float:
        if self.mock:
            return sum(position) / (255 * len(position)) + 0.1

        # Use full train dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        full_train_loader = DataLoader(full_train, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

        layers = []
        for i in range(0, len(position), 2):
            byte0, byte1 = position[i], position[i+1]
            layer = decode_layer(byte0, byte1)
            layers.append(layer)
        layers = [l for l in layers if l.layer_type != LayerType.DISABLED]
        if not layers or layers[-1].layer_type != LayerType.FC:
            layers.append(Layer(LayerType.FC, num_neurons=self.num_classes))
        
        try:
            builder = CNNBuilder(layers, self.num_classes)
            model = builder.build().to(self.device)
            for m in model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
        except Exception:
            return 0.0
            
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for inputs, labels in full_train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return correct / total