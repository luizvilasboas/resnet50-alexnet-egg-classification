import random
import time
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms, models, datasets
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

METRICS_FOLDER = 'metrics'

MODEL_NAME = 'alexnet'

num_classes = 2
class_names = ['damaged', 'not_damaged']
ds_path = 'dataset'

batch_size = 32
lr = 0.001
mm = 0.9
epochs = 50

data_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

full_dataset = datasets.ImageFolder(ds_path, transform=data_transforms)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_size = len(train_dataset)
val_size = len(val_dataset)

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size)

model_ft = models.alexnet(weights='IMAGENET1K_V1')

model_ft.classifier[6] = nn.Linear(4096, num_classes)

model = model_ft

if DEVICE.type == 'cuda':
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mm)

time_total_start = time.time()

train_loss_list = []
train_acc_list = []

val_loss_list = []
val_acc_list = []

for epoch in range(epochs):
    time_epoch_start = time.time()

    model.train() 

    loss_epoch_train = 0.0    
    num_hits_epoch_train = 0  

    for inputs, labels in train_dataloader:

        if DEVICE.type == 'cuda':
            inputs = inputs.to(DEVICE) 
            labels = labels.to(DEVICE) 

        optimizer.zero_grad() 

        torch.set_grad_enabled(True) 

        outputs = model(inputs) 

        preds = torch.argmax(outputs, dim=1).float() 

        loss = criterion(outputs, labels)

        loss.backward() 

        optimizer.step()

        loss_epoch_train += float(loss.item()) * inputs.size(0) 

        num_hits_epoch_train += torch.sum(preds == labels.data) 

    train_loss = loss_epoch_train / train_size

    train_acc = float(num_hits_epoch_train.double() / train_size)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    model.eval()

    loss_epoch_val = 0.0
    num_hits_epoch_val = 0

    for inputs, labels in val_dataloader:
        if DEVICE.type == 'cuda':
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

        optimizer.zero_grad() 

        torch.set_grad_enabled(False) 

        outputs = model(inputs) 

        preds = torch.argmax(outputs, dim=1).float()

        loss = criterion(outputs, labels) 

        loss_epoch_val += float(loss.item()) * inputs.size(0)

        num_hits_epoch_val += torch.sum(preds == labels.data)
        
    val_loss = loss_epoch_val / val_size
    val_acc = float(num_hits_epoch_val.double() / val_size)

    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    time_epoch = time.time() - time_epoch_start
    
    print('Epoch {}/{} - TRAIN Loss: {:.4f} TRAIN Acc: {:.4f} - VAL. Loss: {:.4f} VAL. Acc: {:.4f} ({:.4f} seconds)'.format(epoch, epochs - 1, train_loss, train_acc, val_loss, val_acc, time_epoch))

time_total_train = time.time() - time_total_start

print('\nTreinamento finalizado. ({0}m and {1}s)'.format(int(time_total_train // 60), int(time_total_train % 60)))

epochs_list = []
for i in range(len(train_loss_list)):
    epochs_list.append(i)

loss_title = 'Loss - ' + str(epochs) + ' epochs'
acc_title = 'Accuracy - ' + str(epochs) + ' epochs'

plt.figure()
plt.title(loss_title)
plt.plot(epochs_list, train_loss_list, c='magenta' ,ls='--', label='Train loss', fillstyle='none')
plt.plot(epochs_list, val_loss_list, c='green' ,ls='--', label='Val. loss', fillstyle='none')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.savefig(f'{METRICS_FOLDER}/{MODEL_NAME}_loss.png')

plt.figure()
plt.title(acc_title)
plt.plot(epochs_list, train_acc_list, c='magenta' ,ls='-', label='Train acuracy', fillstyle='none')
plt.plot(epochs_list, val_acc_list, c='green' ,ls='-', label='Val. accuracy', fillstyle='none')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.savefig(f'{METRICS_FOLDER}/{MODEL_NAME}_accuracy.png')

model.eval()
all_labels, all_preds = [], []

for inputs, labels in val_dataloader:
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
    outputs = model(inputs)
    preds = torch.argmax(outputs, dim=1)
    all_labels.extend(labels.cpu().numpy())
    all_preds.extend(preds.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.savefig(f'{METRICS_FOLDER}/{MODEL_NAME}_confusion_matrix.png')

metrics = {
    'precision': precision,
    'recall': recall,
    'f1_score': f1
}

with open(f'{METRICS_FOLDER}/{MODEL_NAME}_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
