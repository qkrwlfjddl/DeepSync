import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import *
import torchvision.transforms as transforms
import torch.nn.init as init

# 필요한 추가 라이브러리들을 여기에 임포트하세요.

from einops import rearrange
from tqdm.auto import tqdm
import wandb
import tyro
from dataclasses import dataclass

torch.manual_seed(0)


@dataclass
class Args:
    wandb_project_name: str = "ViT"
    wandb_run_name: str = "ViT Cifar10 base"
    epochs: int = 200
    learning_rate: int = 0.0001
    batch_size: int = 256
    total_steps: int = int(50000 % batch_size) * epochs
    warmup_steps: int = 4000
    num_class: int = 10
    img_size: int = 32
    patch_size: int = 4
    patch_dim: int = patch_size * patch_size * 3
    num_patch: int = int(img_size / patch_size) ** 2
    num_heads: int = 8
    model_dim: int = 512
    encoder_ffn_dim: int = 1024
    num_blocks: int = 6


args = tyro.cli(Args)

wandb.init(
    project=args.wandb_project_name,
    name=args.wandb_run_name,
    config=vars(args),
    save_code=True,
)
transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),  # Random rotation
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),  # Color jitter
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10
            ),  # Random affine
            transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip
            transforms.RandomPerspective(
                distortion_scale=0.2, p=0.5
            ),  # Random perspective
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
}

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transforms["train"]
)

test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transforms["test"]
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, shuffle=False
)


# 이미지 패치 처리를 위한 클래스
class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # TODO: 이미지 패치 처리 로직 구현
        return rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
            p1=args.patch_size,
            p2=args.patch_size,
        )


# Class Token 추가 클래스
class AddClassToken(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # TODO: Class Token 변수 초기화
        self.cls_token = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        # TODO: Class Token 추가 로직 구현
        batch_size = len(x)
        batch_cls_token = self.cls_token.expand(batch_size, -1, -1)
        batch_cls_token.cuda()
        return torch.cat([batch_cls_token, x], dim=1)


# Positional Embedding 클래스
class PositionalEmbedding(nn.Module):
    def __init__(self, num_seq, patch_dim, d_model):
        super().__init__()
        # TODO: Positional Embedding 변수 초기화
        self.pos_emb = nn.Parameter(torch.randn(num_seq, patch_dim))
        self.mlp = nn.Linear(patch_dim, d_model)
        init.xavier_normal_(self.mlp.weight)

    def forward(self, x):
        # TODO: Positional Embedding 추가 로직 구현
        x += self.pos_emb
        x = self.mlp(x)
        return x


# Transformer Encoder 클래스
class EncoderBlock(nn.Module):
    def __init__(self, num_heads, num_seq, d_model, dff):
        super().__init__()
        self.d_model = d_model
        self.d_k = int(d_model / num_heads)
        self.num_heads = num_heads

        self.q_w = nn.Linear(self.d_model, self.d_model)
        self.k_w = nn.Linear(self.d_model, self.d_model)
        self.v_w = nn.Linear(self.d_model, self.d_model)

        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

        self.layer_norm1 = nn.LayerNorm([num_seq, self.d_model])
        self.layer_norm2 = nn.LayerNorm([num_seq, self.d_model])

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, dff),
            nn.Dropout(0.0),
            nn.GELU(),
            nn.Linear(dff, self.d_model),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        # TODO: Transformer Encoder 로직 구현
        q = self.q_w(x)
        k = self.k_w(x)
        v = self.v_w(x)
        qkv_size = list(q.shape)

        
        # q의 shape(B, seq_len, dim)을 (B, seq_len, head, d_k)로 바꿔줌
        q = q.reshape(qkv_size[:-1]+[self.num_heads, self.d_k])
        k = k.reshape(qkv_size[:-1]+[self.num_heads, self.d_k])
        v = v.reshape(qkv_size[:-1]+[self.num_heads, self.d_k])


        attention = torch.matmul(q.permute(0,2,1,3),k.permute(0,2,3,1)) / torch.sqrt(torch.tensor([self.d_k])).cuda()
        # attention size: (B, h, seq_len, seq_len)임.
        attention = torch.softmax(attention, dim=-1)
        # v: size (B,seq_len,h,d_k)
        attention = torch.matmul(attention,v.permute(0,2,1,3))
        attention = attention.permute(0,2,1,3)
        attention = attention.reshape(qkv_size)


        # Residual Connection
        attention = x + attention
        x = self.layer_norm1(attention)

        # Layer normalization
        xx = self.ffn(x)
        xx = self.dropout2(xx)
        x = x + xx
        x = self.layer_norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_heads, num_seq, d_model, ffn_dim, num_blocks):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(num_heads, num_seq, d_model, ffn_dim)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return x


# Classifier Head 클래스
class ClassifierHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.classify_mlp = nn.Linear(d_model, num_classes)
        init.xavier_normal_(self.classify_mlp.weight)

    def forward(self, x):
        # TODO: Classifier Head 로직 구현
        return self.classify_mlp(x)


# 전체 ViT 모델 조립 클래스
class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = PatchEmbedding()
        self.add_class_token = AddClassToken(dim=args.patch_dim)
        self.positional_embedding = PositionalEmbedding(
            num_seq=args.num_patch + 1, patch_dim=args.patch_dim, d_model=args.model_dim
        )
        self.encoder = Encoder(
            num_heads=args.num_heads,
            num_seq=args.num_patch + 1,
            d_model=args.model_dim,
            ffn_dim=args.encoder_ffn_dim,
            num_blocks=args.num_blocks,
        )
        self.classifier = ClassifierHead(
            d_model=args.model_dim, num_classes=args.num_class
        )

    def forward(self, x):
        # TODO: 전체 모델 순서대로 로직 구현
        x = self.patch_embedding(x)
        x = self.add_class_token(x)
        x = self.positional_embedding(x)
        x = self.encoder(x)
        x = self.classifier(x[:, 0, :])
        return x


# 모델 초기화 및 테스트
model = ViT().cuda()

optimizer = AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    amsgrad=False,
)


scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

criterion = nn.CrossEntropyLoss()


def train(model, trainloader):
    for batch in tqdm(trainloader, desc="train", leave=False):
        data, label = batch
        data = data.cuda()
        label = label.cuda()
        optimizer.zero_grad()

        pred = model(data)
        loss = criterion(pred, label)
        wandb.log({"loss": loss})
        pred = pred.argmax(dim=1)
        num_samples = label.size(0)
        num_correct = (pred == label).sum()
        wandb.log({"train_acc": (num_correct / num_samples * 100).item()})
        loss.backward()
        optimizer.step()
    scheduler.step()


@torch.inference_mode()
def evaluate(model, testloader):
    model.eval()
    num_samples = 0
    num_correct = 0
    for batch in tqdm(testloader, desc="eval", leave=False):
        data, label = batch
        data = data.cuda()
        label = label.cuda()

        pred = model(data)
        pred = pred.argmax(dim=1)
        num_samples += label.size(0)
        num_correct += (pred == label).sum()

    return (num_correct / num_samples * 100).item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = args.epochs

for epoch_num in tqdm(range(1, EPOCHS + 1)):
    train(model, train_loader)
    metric = evaluate(model, test_loader)
    wandb.log({"test_acc": metric})
    if epoch_num % 50 == 0:
        torch.save(
            {
                "epoch": epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"model{epoch_num}.pt",
        )

