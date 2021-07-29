import random
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

from model_VGG import vgg19, vgg19_bn

import config

def cos_sim(a, b):
    return F.cosine_similarity(a, b)

def infer():
    #초기화
    random.seed(config.seed_num)
    torch.manual_seed(config.seed_num)
    torch.cuda.manual_seed_all(config.seed_num)

    cuda = torch.cuda.is_available()

    device = torch.device('cuda' if cuda else 'cpu')

    #model 정의
    model = vgg19_bn()
    model.load_state_dict(torch.load(config.pth_FilePATH+config.infModelName_to_load+".pth"))
    model = model.to(device)

    #data 불러오기
    submission = pd.read_csv("../sample_submission.csv")

    left_test_paths = list()
    right_test_paths = list()

    for i in range(len(submission)):
        left_test_paths.append(submission['face_images'][i].split()[0])
        right_test_paths.append(submission['face_images'][i].split()[1])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    #왼쪽 이미지 
    left_test = list()

    for left_test_path in left_test_paths:
        
        img = Image.open(config.testDataPATH + left_test_path + '.jpg').convert("RGB")# 경로 설정 유의(ex .inha/test)
        img = data_transform(img) # 이미지 데이터 전처리
        left_test.append(img) 
    
    left_test = torch.stack(left_test)

    left_infer_result_list = list()

    with torch.no_grad():
        '''
        메모리 부족으로 6,000개 (배치) 한번에 입력으로 넣지 않고 1,000개 씩 입력으로 줌
        '''
        batch_size = 1000
        for i in range(0, 6):
            i = i * batch_size
            tmp_left_input = left_test[i:i+batch_size]
            #print(tmp_left_input.size()) # torch.Size([1000, 3, 112, 112]) -> torch.Size([1000, 3, 112, 112])
            
            # 출력 (라벨,라벨 직전 벡터) --> 왼쪽 오른쪽 벡터로 구분
            _, left_infer_result = model(tmp_left_input.to(device))
            #print("left")
            #print(left_infer_result.size()) # torch.Size([1000, 512])   ->  torch.Size([1000, 512, 7, 7]) --flatten--> torch.Size([1000, 25088])
            left_infer_result_list.append(left_infer_result)

        left_infer_result_list = torch.stack(left_infer_result_list, dim=0).view(-1, 25088)  #512 -> 25088
        #print("left")
        #print(left_infer_result_list.size()) # torch.Size([6000, 512]) --> torch.Size([294000, 512])

    #오른쪽 이미지 
    right_test = list()
    for right_test_path in right_test_paths:
        img = Image.open(config.testDataPATH + right_test_path + '.jpg').convert("RGB") # 경로 설정 유의 (ex. inha/test)
        img = data_transform(img)# 이미지 데이터 전처리
        right_test.append(img)
    right_test = torch.stack(right_test)
    #print(right_test.size()) # torch.Size([6000, 3, 112, 112])

    right_infer_result_list = list()
    with torch.no_grad():
        '''
        메모리 부족으로 6,000개 (배치) 한번에 입력으로 넣지 않고 1,000개 씩 입력으로 줌
        '''
        batch_size = 1000
        for i in range(0, 6):
            i = i * batch_size
            tmp_right_input = right_test[i:i+batch_size]
            #print(tmp_input.size()) # torch.Size([1000, 3, 112, 112])
            _, right_infer_result = model(tmp_right_input.to(device))
            #print("right")
            #print(right_infer_result.size()) # torch.Size([1000, 512]) -->torch.Size([1000, 25088])
            right_infer_result_list.append(right_infer_result)



        right_infer_result_list = torch.stack(right_infer_result_list, dim=0).view(-1, 25088) #512 -> 25088
        #print("right")
        #print(right_infer_result_list.size()) # torch.Size([6000, 512]) --> torch.Size([294000, 512])

    cosin_similarity = cos_sim(left_infer_result_list, right_infer_result_list)
    
    # 최종
    submission = pd.read_csv("../sample_submission.csv") 
    submission['answer'] = cosin_similarity.tolist()
    #submission.loc['answer'] = submission['answer']
    submission.to_csv(config.submissionName, index=False)

if __name__ == '__main__':
    infer()