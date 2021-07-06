from torchsummary import summary
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm
from SemiSupervise import get_pseudo_labels
from StudentNetwork import *
import sys
sourcePath = "/tmp2/b08902011/"

train_tfm = transforms.Compose([
    transforms.Resize((142, 142)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(128),
    transforms.ToTensor(),
])

test_tfm = transforms.Compose([
    transforms.Resize((142, 142)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])

batch_size = 64
train_set = DatasetFolder(sourcePath + "food-11/training/labeled",
                          loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder(
    "food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder(sourcePath + "food-11/training/unlabeled",
                              loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder(
    sourcePath + "food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size,
                          shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)



student = Student()
student_net = student.network
summary(student_net, (3, 128, 128), device="cpu")


def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.5, T=2.33):
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    soft_loss = T * T * alpha * nn.KLDivLoss()(nn.Softmax(-1)(teacher_outputs/T),
                                       nn.Softmax(-1)(outputs/T))
    return hard_loss + soft_loss


teacher_net = torch.load(sourcePath + 'teacher_net.ckpt')
teacher_net.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
student_net = student_net.to(device)
teacher_net = teacher_net.to(device)
unlabeled_set = get_pseudo_labels(unlabeled_set, teacher_net)
concat_dataset = ConcatDataset([train_set, unlabeled_set])
train_loader = DataLoader(
    concat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
criterion = nn.CrossEntropyLoss()
optimizer = student.optimizer
n_epochs = 80
if '-l' in sys.argv:
    student.load(sourcePath)
    student_net = student.network
    optimizer = student.optimizer
for epoch in range(n_epochs):
    student_net.train()
    train_loss = []
    train_accs = []
    for batch in tqdm(train_loader):
        imgs, labels = batch
        logits = student_net(imgs.to(device))
        with torch.no_grad():
            soft_labels = teacher_net(imgs.to(device))
        loss = loss_fn_kd(logits, labels.to(device), soft_labels)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            student_net.parameters(), max_norm=10)
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print(
        f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    student.save(sourcePath)
    # ==============================Validation============================================
    valid_loss = []
    valid_accs = []
    student_net.eval()
    for batch in tqdm(valid_loader):
        imgs, labels = batch
        with torch.no_grad():
            logits = student_net(imgs.to(device))
            soft_labels = teacher_net(imgs.to(device))
        loss = loss_fn_kd(logits, labels.to(device), soft_labels)
        acc = (logits.argmax(dim=-1) == labels.to(device)
               ).float().detach().cpu().view(-1).numpy()
        valid_loss.append(loss.item())
        valid_accs += list(acc)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    print(
        f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

#=================Prediction===========================
predictions = []
student_net.eval()
for batch in tqdm(test_loader):
    imgs, _ = batch
    with torch.no_grad():
        logits = student_net(imgs.to(device))
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
#https://www.kaggle.com/c/ml2021spring-hw13/leaderboard
with open("predict.csv", "w") as f:
    f.write("Id,Category\n")
    for i, pred in enumerate(predictions):
        f.write(f"{i},{pred}\n")
