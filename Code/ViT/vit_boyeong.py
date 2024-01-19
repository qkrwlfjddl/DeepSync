import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from einops import rearrange
import torch.nn.functional as F

# 데이터 로더 및 배치 설정
# TODO: 데이터셋과 DataLoader 설정
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)


# 이미지 패치 처리를 위한 클래스
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=48):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        patches = self.projection(x)
        patches = rearrange(patches, 'b c h w -> b (h w) c')
        return patches


# Class Token 추가 클래스
class AddClassToken(nn.Module):
    def __init__(self, dim, emb_size, batch_size=128):
        super().__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x):
        cls_tokens = self.cls_token.expand(self.batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=self.dim)
        return x


# Positional Embedding 클래스
class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, dim=768, emb_size=48):
        super().__init__()
        self.embedding = nn.Linear(emb_size, dim)
        self.positions = nn.Parameter(torch.randn(1, seq_len, emb_size))

    def forward(self, x):
        x = x + self.positions
        x = self.embedding(x)
        return x


# Transformer Encoder 클래스
class ViTTransformer(nn.Module):
    def __init__(self, d_model=768, num_heads=8, ff_hidden_dim=2048):
        super().__init__()
        self.d_k = d_model // num_heads
        self.d_kroot = self.d_k ** 0.5
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc1_w0 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])
        self.fc1 = nn.Linear(d_model, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        for _ in range(4):
            orinx = x
            x = self.norm1(x)
            Q = self.query(x)
            K = self.key(x)
            V = self.value(x)
            Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.d_k)
            K = rearrange(K, 'b n (h d) -> b h n d', h=self.d_k)
            V = rearrange(V, 'b n (h d) -> b h n d', h=self.d_k)
            attscore = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.d_kroot
            attentionD = F.softmax(attscore, dim=-1)
            attentionV = torch.matmul(attentionD, V)
            attm = rearrange(attentionV, 'b h n d -> b n (h d)')
            attf = self.fc1_w0(attm)
            attfnorm = orinx + self.dropout(attf)
            attfnormplus = self.norm2(attfnorm)
            attfnormffn = F.relu(self.fc1(attfnormplus))
            attfnormffn = self.fc2(attfnormffn)
            x = attfnorm + self.dropout(attfnormffn)
        return x


# Classifier Head 클래스
class ClassifierHead(nn.Module):
    def __init__(self, num_classes=10, in_dim=768):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = x[:, 0, :]  # Extract Class Token
        x = self.linear(x)
        return x


# 전체 ViT 모델 조립 클래스
class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = PatchEmbedding()
        self.add_class_token = AddClassToken(dim=1, emb_size=48)
        self.positional_embedding = PositionalEmbedding(seq_len=65)
        self.transformer = ViTTransformer()
        self.classifier = ClassifierHead()

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.add_class_token(x)
        x = self.positional_embedding(x)
        x = self.transformer(x)
        output = self.classifier(x)
        return output


# 모델 초기화 및 테스트
model = ViT()
# 모델 훈련 부분

# 손실 함수
# TODO: 적절한 손실 함수 설정

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochn = 10
for epoch in range(epochn):
    model.train()

    for img, labels in train_loader:
        optimizer.zero_grad()

        # 입력 데이터에 패딩 마스크 추가
        # mask = torch.ones(img.size(0), 1, 65)  # 패딩 마스크 예시 (모든 위치가 패딩이 아닌 경우)
        outputs = model(img)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 각 에폭마다 손실 출력
    print(f'Epoch [{epoch + 1}/{epochn}], Loss: {loss.item()}')

test_input = torch.rand(1, 3, 32, 32)  # 예시 입력
test_output = model(test_input)
print(test_output.shape)  # 예상 출력: [1, 10]
