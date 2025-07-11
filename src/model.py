import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,in_channels=3,out_channels=8):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.SEQ=nn.Sequential(#IMG_size= [3, 208, 70] -> input: [batch_size, 3, 208, 70]
            nn.Conv2d(in_channels=in_channels,out_channels=16,kernel_size=3,stride=1,padding=1),  # batch_size, 32, 208, 70
            nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),   # batch_size, 32, 104, 35
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),  # batch_size, 64, 104, 35
            nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),   # batch_size, 64, 52, 17
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1), # batch_size, 128,  52, 17
            nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),  # batch_size, 128, 26, 8
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),  # batch_size, 256, 26, 8
            nn.BatchNorm2d(128),nn.ReLU(),  # batch_size, 256, 26, 8
            nn.AdaptiveAvgPool2d((1, 1))    # batch_size, 256, 1, 1
        )
        self.fc=nn.Sequential(nn.Flatten(),
                              nn.Linear(in_features=128,out_features=512),
                              nn.ReLU(),nn.Dropout(0.3),
                              nn.Linear(in_features=512,out_features=31)) # 31 classes 
    def forward(self,x):
        x=self.SEQ(x)
        x=self.fc(x)
        return x

# transform=transforms.Compose([transforms.ToTensor()])
# img_path='/home/chu-tung/Desktop/Deep_learning/LICENSE_PLATE_PIPELINE/Predict_Char/data/processed/0/0_0000_02187_b.jpg'
# img=cv2.imread(img_path)
# img=transform(img)
# batch=torch.unsqueeze(img,0)
# print(img.shape)
# print(batch.shape)
# model=CNN()
# # from torchinfo import summary
# # summary(model, input_size=(1, 3, 208, 70))
# # print(summary)
# out=model.forward(batch)
# print(out.shape)
# summary(model, input_size=(1, 3, 150, 35))
# print(summary)