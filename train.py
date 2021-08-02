import time
import random
import datetime

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR , StepLR

from dataset import MS1MDataset
from model_RESNET import ResNet, ResNet_Final ,IRBlock

import config
import shutil

def model_train(model, train_loader, optimizer, criterion, scheduler, total_step, device):
    model.train()

    running_loss = 0 
    running_corrects = 0
    total_num = 0

    start_time = time.time()
    total_batch_num = len(train_loader)
    for i, data in enumerate(train_loader):
        total_step += 1
        inputs, labels = data

        batch_size = inputs.size(0)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        output, features = model(inputs) 
               
        _, preds = torch.max(output, 1)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 300 == 0:
            lr = scheduler.get_lr()[0]
            '''
            for param_group in optimizer.param_groups: 
                lr = param_group['lr']
            '''
            print('{} lr: {:7f}, train_batch: {:4d}/{:4d}, loss: {:.4f}, acc: {:.4f}, time: {:.2f}'
                  .format(datetime.datetime.now(), lr, i, total_batch_num, loss.item(), torch.sum(preds == labels.data).item() / batch_size, time.time() - start_time))
            start_time = time.time()

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item() / batch_size
        
    epoch_loss = running_loss / (total_batch_num)
    epoch_acc = running_corrects / (total_batch_num)
    
    return epoch_loss, epoch_acc

def model_eval(model, test_loader, criterion, device):
    model.eval()

    running_loss = 0 
    running_corrects = 0
    total_num = 0

    start_time = time.time()
    total_batch_num = len(test_loader)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            batch_size = inputs.size(0)

            output, features = model(inputs) 
                
            _, preds = torch.max(output, 1)
            loss = criterion(output, labels)

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item() / batch_size
        
    epoch_loss = running_loss / (total_batch_num)
    epoch_acc = running_corrects / (total_batch_num)
    
    return epoch_loss, epoch_acc

def main():
    
    random.seed(config.seed_num)
    torch.manual_seed(config.seed_num)
    torch.cuda.manual_seed_all(config.seed_num)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    gpu_num = torch.cuda.device_count()
    #-------------------------- Model Initialize --------------------------

    res_model = ResNet(IRBlock, [3, 4, 6, 3], use_se=True, im_size=112)
    net = nn.Sequential(nn.Linear(512, config.num_classes))

    model = ResNet_Final(res_model, net)
    
    if config.Load_Model :
        model.load_state_dict(torch.load(config.pth_FilePATH+config.ModelName_to_load+".pth"))

    model = model.to(device)
    #-------------------------- Loss & Optimizer --------------------------
    criterion = nn.CrossEntropyLoss()
    
    if gpu_num > 1:
        print("DataParallel mode")
        model = nn.DataParallel(model).to(device)
        optimizer = optim.Adam(model.module.parameters(), lr=0.001)

        #step에서 바뀜 
        #lr_lambda = lambda x: 1 if x < 1000 else (x / 1000) ** -0.5
        #scheduler = LambdaLR(optimizer, lr_lambda)

        #코사인
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)

        #스텝
        #scheduler = StepLR(optimizer, step_size=10, gamma=0.5)


    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        #lr_lambda = lambda x: x/1000 if x < 1000 else (1 if x < 20000 else (x / 20000) ** -0.5 )
        #scheduler = LambdaLR(optimizer, lr_lambda)
    
    #-------------------------- Data load --------------------------
    #train dataset
    #자기 파일 path
    train_dataset = MS1MDataset(config.trainDataPATH , config.trainDataListPATH)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, config.batch_size, config.train_data_shuffle, num_workers=gpu_num * 4)

    #test데이터 셋을 train 폴더에서 가져오므로 폴더 위치는 trainDataPATH!!!!!
    test_dataset = MS1MDataset(config.trainDataPATH, config.testDataListPATH) # 폴더 위치 , 파일 리스트
    test_dataloader = torch.utils.data.DataLoader(test_dataset, config.batch_size, config.test_data_shuffle, num_workers=gpu_num * 4)


    # ----------------------- 학습 시작 ----------------------------------
    print(" ")
    print("학습시작")
    print(" ")

    # 현재 config 저장
    config_log_PATH = config.configs_path+ "config_"+ config.trainName +".py"
    shutil.copy("./config.py",config_log_PATH)


    pre_test_acc = 0
    pre_test_loss = 100000
    total_step = 0
    for epoch in range(0, config.MaxEpoch):
        
        print('{} 학습 시작'.format(datetime.datetime.now()))
        train_time = time.time()
        epoch_loss, epoch_acc = model_train(model, train_dataloader, optimizer, criterion, scheduler, total_step, device)
        train_total_time = time.time() - train_time
        print('{} Epoch {} (Training) Loss {:.4f}, ACC {:.4f}, time: {:.2f}'.format(datetime.datetime.now(), epoch+1, epoch_loss, epoch_acc, train_total_time))
        
        print('{} 평가 시작'.format(datetime.datetime.now()))
        eval_time = time.time()
        test_epoch_loss, test_epoch_acc = model_eval(model, test_dataloader, criterion, device)
        eval_total_time = time.time() - eval_time
        print('{} Epoch {} (eval) Loss {:.4f}, ACC {:.4f}, time: {:.2f}'.format(datetime.datetime.now(), epoch+1, test_epoch_loss, test_epoch_acc, eval_total_time))
        

        # model 저장        
        if test_epoch_acc > pre_test_acc:
            print("best model을 저장하였습니다.")
            if gpu_num > 1:
                torch.save(model.module.state_dict(), config.pth_FilePATH + config.trainName + ".pth")
            else:
                torch.save(model.state_dict(), config.pth_FilePATH + config.trainName + ".pth")
            pre_test_acc = test_epoch_acc

        # 에폭별 저장 
        """
        if gpu_num > 1:
            torch.save(model.module.state_dict(), config.pth_FilePATH + config.trainName +"_"+ str(epoch) + ".pth")
        else:
            torch.save(model.state_dict(), config.pth_FilePATH + config.trainName +"_"+ str(epoch) + ".pth")
        """
        
        # 10에폭 마다 모델 저장 
        if (epoch%10 == 0):
            print("model every 10..")
            torch.save(model.module.state_dict(), config.pth_FilePATH + config.trainName +"_"+ str(epoch) + ".pth")


        # log 저장
        print("Log ...")
        with open(config.train_log_path+"TrainLog_"+ config.trainName +".txt", "a") as ff:
            ff.write('Epoch %d (Training) Loss %0.4f Acc %0.4f time %0.4f' % (epoch+1, epoch_loss, epoch_acc, train_total_time))
            ff.write('\n')
            ff.write('Epoch %d (val) Loss %0.4f Acc %0.4f time %0.4f ' % (epoch+1,test_epoch_loss, test_epoch_acc, eval_total_time))
            ff.write('\n')
            ff.write('\n')
        
if __name__ == '__main__':
    main()