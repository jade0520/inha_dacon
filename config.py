# DEFAULT
seed_num = 123456
num_classes = 42711

#Setting

#Pre-trained
Load_Model = True
ModelName_to_load = "model_VGG_TEST3_2"

trainName = "model_VGG_TEST3" # CosineAnnealingLR , MAxPooling2d ->1,
MaxEpoch = 100 # 테스트 할때는 에폭 2로만

## DataLoader
batch_size = 256 #512
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
infModelName_to_load  = "model_VGG_TEST3"

## 추론 데이터 path
testDataPATH= "../test/"
submissionName = "../"+infModelName_to_load+".csv"