## Libraries
import numpy as np
import os, torch, argparse, sys
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler, SubsetRandomSampler
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from networks.Model_refine import ResNet_Model
from networks.triplet.triplet_attention_resnet import ResidualNet as Triplet_Model
from networks.cbam.cbam_resnet import ResidualNet as CBAM_Model
from networks.gct.gct_resnet import ResidualNet as GCT_Model
from networks.non_local_neural_networks.nln_resnet import ResidualNet as NLN_Model
from networks.proposed.Con_Att_resnet import ResidualNet as Con_Att_Model

from datasets_github2 import PCB_Dataset
from losses import ContrastiveLoss_yl
from train import train_model

# Argument Parser 설정
parser = argparse.ArgumentParser(description="Train and evaluate models for PCB defect detection")
parser.add_argument('-m', '--model', type=str, choices=['resnet18', 'proposed', 'cbam', 'nln', 'gct', 'triplet'], default='proposed')
parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
parser.add_argument('-bs', '--batch_size', type=int, default=64)
parser.add_argument('-e', '--epochs', type=int, default=5)
parser.add_argument('-c', '--control', type=float, default=0.00001)
parser.add_argument('-ma', '--margin', type=float, default=1.0)

args = parser.parse_args()

# 프로젝트의 기본 경로와 데이터 경로 설정
project_path = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 절대 경로
data_path = os.path.join(project_path, '..', 'data')  # data 폴더 경로 설정 (src 폴더 기준 상위 디렉토리)

# 데이터 디렉토리가 존재하지 않으면 에러 메시지를 출력하고 종료합니다.
if not os.path.exists(data_path):
    raise FileNotFoundError(f"데이터가 {data_path} 경로에 없습니다. 데이터를 'data/' 디렉토리에 넣어주세요.")

# PCB_Dataset 클래스 사용 예시 (datasets_github2.py에서 정의한 클래스)
Train_dataset = PCB_Dataset(data_path=data_path, Train=True)
Valid_dataset = PCB_Dataset(data_path=data_path, Train=False)
Test_dataset = PCB_Dataset(data_path=data_path, Train=False)

# 레이블 세트 생성
label_set = np.array([Train_dataset.class_to_idx[data[1]] for data in Train_dataset.data])

### 데이터 샘플링 및 비율 조절 ###

temp_data, test_data_idx = train_test_split(range(len(Train_dataset)), test_size=0.2, stratify=label_set)
train_data_idx, valid_data_idx = train_test_split(temp_data, stratify=label_set[temp_data], test_size=0.25)

# 클래스 가중치 계산
_, class_counts = np.unique(label_set, return_counts=True)
num_samples = sum(class_counts)

class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
class_weights[1] *= (1 / 3)  # 특정 클래스에 대한 가중치 조정

# 샘플 가중치 부여 및 샘플러 생성
sample_weights = [class_weights[label_set[i]] for i in train_data_idx]
weightedsampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights))

# DataLoader 생성 (학습용, 검증용, 테스트용)
Train_dataset2 = Subset(Train_dataset, train_data_idx)
valid_sampler = SubsetRandomSampler(valid_data_idx)
test_sampler = SubsetRandomSampler(test_data_idx)

train_dataLoader = DataLoader(Train_dataset2, sampler=weightedsampler, shuffle=False, batch_size=args.batch_size)
valid_dataLoader = DataLoader(Valid_dataset, sampler=valid_sampler, shuffle=False, batch_size=args.batch_size)
test_dataLoader = DataLoader(Test_dataset, sampler=test_sampler, shuffle=False, batch_size=args.batch_size)

# Device 설정 (CUDA 사용 여부 확인)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


Train_Valid_DataLoader_Set = {'train': train_dataLoader, 'val': valid_dataLoader}

# 모델 선택 및 초기화
if args.model == 'resnet18':
    model = ResNet_Model(num_layer=18).cuda()
elif args.model == 'proposed':
    model = Con_Att_Model(18, 2, use_att=True).cuda()
elif args.model == 'cbam':
    model = CBAM_Model('ImageNet', 18).cuda()
elif args.model == 'nln':
    model = NLN_Model(18).cuda()
elif args.model == 'gct':
    model = GCT_Model(18).cuda()
elif args.model == 'triplet':
    model = Triplet_Model(18).cuda()
    
model = torch.nn.DataParallel(model)  # 다중 GPU 지원

# 손실 함수 및 옵티마이저 설정
criterion1 = torch.nn.CrossEntropyLoss()
criterion2 = ContrastiveLoss_yl(args.margin)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                lr_lambda=lambda epoch: 0.99 ** epoch,
                                                last_epoch=-1,
                                                verbose=False)

# 모델 학습 함수 호출 (train_model 함수는 외부에서 가져옴)
(history_set, history_set2, history_set3,
    end_time,
    epoch_train_time,
    epoch_valid_time) = train_model(Train_Valid_DataLoader_Set,
                                model,
                                criterion1,
                                criterion2,
                                optimizer,
                                scheduler,
                                args,
                                device,
                                total_loss=True,
                                control=args.control,
                                num_epochs=args.epochs)




## 테스트 데이터에 대한 예측 ##
print("테스트 데이터에 대한 예측을 시작합니다...")

model.eval()  # 평가 모드 전환

targets = []
predictions = []

with torch.no_grad():  # 그래디언트 계산 비활성화 (평가 시 필요 없음)
    for current_img, master_img, _, label in test_dataLoader:
        current_img = current_img.to(device)
        master_img = master_img.to(device)
        label = label.to(device)

        # 모델 예측 수행
        if args.model == "proposed":
            outputs, _, _ = model(current_img, master_img)
        
        else: 
            outputs, _ = model(current_img)
        
        _, predicted = torch.max(outputs.data, 1)

        targets.extend(label.cpu().numpy())
        predictions.extend(predicted.cpu().numpy())

# 성능 출력 (classification_report 사용)
from sklearn.metrics import classification_report
print("테스트 결과:")
print(classification_report(targets, predictions))