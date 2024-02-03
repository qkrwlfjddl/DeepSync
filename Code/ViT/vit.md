## 1.데이터 로더 및 배치 설정:

PyTorch의 DataLoader를 사용하여 Cifar-10 데이터셋을 불러옵니다. 여기서 batch_size를 설정하여 학습에 사용할 배치의 크기를 정의합니다.

## 2.이미지 패치 처리:

입력 이미지(Cifar-10, 32x32x3)를 4x4 패치로 나눕니다. 이렇게 하면 총 64개의 패치가 생성됩니다.
각 패치는 4x4x3 = 48 크기의 벡터로 평탄화됩니다. 이때, `einops.rearrange()` 라이브러리를 사용하면 편리합니다. 결과적으로 입력 이미지는 (batch_size, 64, 48) 형태의 텐서로 변환됩니다.

## 3.Class Token 추가:

Class Token을 입력 시퀀스의 맨 앞에 추가합니다. 이 토큰은 분류 작업을 위한 것으로, 전체 패치 시퀀스를 대표합니다.
이후 입력 시퀀스의 크기는 (batch_size, 65, 48)이 됩니다.

## 4. Positional Embedding:

Learnable Positional Embedding을 각 패치에 추가합니다. Positional Embedding의 크기는 (1, 65, 48)입니다.

입력 시퀀스를 d_model=512 차원의 임베딩으로 변환합니다. 이때, nn.Linear를 사용하면 편리합니다. 이 과정을 통해 입력 시퀀스는 (batch_size, 65, d_model)의 크기를 가지게 됩니다.

## 5. Transformer Encoder:

입력 시퀀스는 Transformer Encoder를 통과합니다. 여기서 d_model=512과 8개의 attention head(num_heads=8)를 사용합니다.
Multi-Head Attention: 각 head의 크기는 d_k = d_model / num_heads = 64이 됩니다.
Feed Forward Network (FFN): 두 개의 linear layer로 구성되며, 크기는 각각 (512, 2048)과 (2048, 256)입니다.
각 sub-layer(attention과 FFN) 뒤에는 residual connection과 Layer Normalization이 적용됩니다.
이 과정은 4번 반복됩니다.

## 6. Classifier Head

Transformer Encoder의 출력에서 Class Token에 해당하는 부분만을 추출합니다. 이 토큰은 (batch_size, 768)의 크기를 가집니다.
이를 linear layer를 통과시켜 최종 출력 크기를 (batch_size, 10)으로 만듭니다. 이는 Cifar-10의 10개 클래스에 대응합니다.

## 7. 손실 함수:

모델의 출력과 실제 레이블 간의 cross entropy loss를 계산합니다.
