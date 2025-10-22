from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matpltlib

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

#-------------------#
#	MLPモデル構築	#
#-------------------#


#nn.Module ... Pytorchの神クラス。
#自分のニューラルネットワークを作成する際は必ずnn.Moduleを継承して作成する。
#		def __init__(self):
#		これは親クラスの初期化である。
#		ネットワーク構成をSequentialで一気に定義する。
#		順番に処理が流れるようになる。
#| 層                  | 説明                        |
#| ------------------ | ------------------------- |
#| `Flatten()`        | 画像(1×28×28)を1次元(784次元)に変換 |
#| `Linear(784, 256)` | 入力層 → 隠れ層1                |
#| `ReLU()`           | 活性化関数                     |
#| `Linear(256, 128)` | 隠れ層2                      |
#| `ReLU()`           | 活性化関数                     |
#| `Linear(128, 10)`  | 出力層（10クラス＝数字0〜9）          |


class MLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Flatten(),
			nn.Linear(28*28, 256),
			nn.ReLU(),
			nn.Linear(256,128),
			nn.ReLU(),
			nn.Linear(128,10)
		)
	def forward(self, x):
		return self.layers(x)
model = MLP()

#-------------------#
#	CNNモデル構築	#
#-------------------#

#	組み込み層ブロック
# self.conv = nn.Sequential(
#    nn.Conv2d(1, 32, 3, 1),   # 入力1チャンネル → 出力32チャンネル、カーネル3x3
#    nn.ReLU(),                # 活性化関数
#    nn.MaxPool2d(2),          # 特徴マップを2x2で縮小
#    nn.Conv2d(32, 64, 3, 1),  # 32→64チャンネル
#    nn.ReLU(),
#    nn.MaxPool2d(2)
#)

#	結合層ブロック
# self.fc = nn.Sequential(
#    nn.Flatten(),
#    nn.Linear(64*5*5, 128),
#    nn.ReLU(),
#    nn.Linear(128, 10)
#)

#	順伝搬
#def forward(self, x):
#    x = self.conv(x)  # 畳み込みブロック
#    x = self.fc(x)    # 全結合ブロック
#    return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = CNN()

#-----------------------#
#		学習ループ		#
#-----------------------#

import torch.utils.data as data

train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=1000)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done.")

#-----------------------#
#		精度評価			#
#-----------------------#

correct = 0
total = 0
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"Accuracy: {correct / total * 100:.2f}%")
