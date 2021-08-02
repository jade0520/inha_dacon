# DEFAULT
seed_num = 123456
num_classes = 42711

#Load Model
Load_Model = False
ModelName_to_load = "model_SEResNet_TEST"

#Setting
trainName = "model_SEResNet_TEST" # lr 크게, 레이어 추가 ,lr scheduler   CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001), 시작 0005로
MaxEpoch = 100 # 테스트 할때는 에폭 2로만

## DataLoader
batch_size = 512
train_data_shuffle = True
test_data_shuffle = False

# PATH
## trained models
pth_FilePATH = "./saved_models/"

## data Path
trainDataPATH = "../train/"
trainDataListPATH = "./train.txt"
testDataListPATH = "./test.txt"

## log
train_log_path = "./log/train_log/"
configs_path = "./log/configs/"


#----------------------------------Inference------------------------------

##추론시 불러올 모델 이름
infModelName_to_load  = "model_SEResNet_TEST"

## 추론 데이터 path
testDataPATH= "../test/"
submissionName = "../"+infModelName_to_load+".csv"