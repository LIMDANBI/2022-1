import torch
import torchsummary
from torch import nn
from torch import optim
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import path_to_img
from model import RobustModel

import pandas as pd

if __name__ == '__main__':
    
    # ---------- 파라미터 설정 ------------
    start_epoch = 1
    epoch_num = 15
    batch_size = 256
    lr = 0.001
    img_size = 28
    
    train_loss_list = []
    val_loss_list = []
    
    
    # ---------- gpu 설정 ------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print()
    
    
    # ---------- model 생성 ------------
    model = RobustModel()
    model.to(device)
    torchsummary.summary(model, (3, img_size, img_size)) # model 정보 
    print()
    
    
    # ---------- 최적화 기법 및 손실 함수 ------------
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    
    # ---------- 이미지 변형 ------------
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    
    train_data = pd.read_csv('./data/train.csv')
    valid_data = pd.read_csv('./data/valid.csv')
    
    train_X = train_data['img_path']  # img_path
    train_y = train_data['label']  # label
    
    valid_X = valid_data['img_path']  # img_path
    valid_y = valid_data['label']  # label
    
    train_datasets = path_to_img(img_path=train_X, labels=train_y, transform=transform)
    valid_datasets = path_to_img(img_path=valid_X, labels=valid_y, transform=transform)

    # ---------- data loader 생성 ------------
    train_loader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_datasets, batch_size=batch_size)

    # ---------- train ------------
    for epoch in range(start_epoch, epoch_num+1):
        print('[epoch %d]' % epoch)
        
        train_loss = 0.0
        train_acc = 0.0
        train_total = 0
        
        valid_loss = 0.0
        valid_acc = 0.0
        valid_total = 0
        
        # 학습
        model.train()
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            
            out = model(img)
            _, predicted = torch.max(out, 1) 
            
            loss = criterion(out, label)
            train_loss = train_loss + loss.item()
            train_total += out.size(0) 
            train_acc += (predicted == label).sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_acc = train_acc / train_total
        avg_train_loss = train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        
        print('train loss : %.4f || train accuracy: %.4f' % (avg_train_loss, train_acc))
        
    
        # ---------- validation ------------
        # 검증
        model.eval()
        with torch.no_grad():
            for img, label in valid_loader:

                img = img.to(device)
                label = label.to(device)

                out = model(img)
                _, predicted = torch.max(out, 1) 

                loss = criterion(out, label)
                valid_loss = valid_loss + loss.item()
                valid_total += out.size(0) 
                valid_acc += (predicted == label).sum()

            valid_acc = valid_acc / valid_total
            avg_val_loss = valid_loss / len(valid_loader)
            val_loss_list.append(avg_val_loss)

            print('valid loss : %.4f || valid accuracy: %.4f' % (avg_val_loss , valid_acc))
        
        # 최적의 모델 저장 early stopping
        if epoch<2:
            min_loss = val_loss_list[-1]
            print('first model save...')
            torch.save(model.state_dict(), '/home/danbibibi/jupyter/Project/model.pt')
        else:
            if val_loss_list[-1] <= min_loss and val_loss_list[-1]<train_loss_list[-1]:
                min_loss = val_loss_list[-1]
                print('better model save...')
                torch.save(model.state_dict(), '/home/danbibibi/jupyter/Project/model.pt')
            else:
                print()
                print('---------- Early Stopping ----------')
                print()
                break
        print()