# DEFAULT
seed_num = 123456
num_classes = 42711

#Setting
Load_Model = False
trainName = "model_VGG"
MaxEpoch = 2

## DataLoader
batch_size = 256
train_data_shuffle = True
test_data_shuffle = False

# PATH
## trained models
pth_FilePATH = "./saved_models/"

### 훈련시 불러올 모델 이름
ModelName_to_load = "model_best"

## data Path
trainDataPATH = "../train/"
trainDataListPATH = "./train.txt"
testDataListPATH = "./test.txt"

## log
train_log_path = "./log/train_log/"
configs_path = "./log/configs/"


# Inference

##추론시 불러올 모델 이름
infModelName_to_load  = "model_test"

## 추론 데이터 path
testDataPATH= "../test/"