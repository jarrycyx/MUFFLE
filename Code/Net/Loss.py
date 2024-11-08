import torch.nn
import torch
import numpy as np

class ColorLoss(torch.nn.Module):
    def __init__(self, basicLossFunc=torch.nn.MSELoss()):
        super().__init__()
        self.basicLossFunc = basicLossFunc
    
    def colorLoss(self, inputImg, target):
        cosEbLoss = torch.nn.CosineEmbeddingLoss(reduction="mean")
        return cosEbLoss(inputImg.view(inputImg.shape[0],-1), target.view(inputImg.shape[0],-1), torch.ones(1).to(inputImg.device))
                
         
    def forward(self, inputImg, target, colorRatio=0.4):
        basicLoss = self.basicLossFunc(inputImg, target)
        colorLoss = self.colorLoss(inputImg, target)
        
        return basicLoss*(1-colorRatio) + colorLoss*colorRatio


class WeightedLoss(torch.nn.Module):
    def __init__(self, basicLossFunc=torch.nn.MSELoss(), gamma=1):
        super().__init__()
        self.gamma = gamma
        self.basicLossFunc = basicLossFunc

    def forward(self, inputImg, target):
        weight = target ** self.gamma
        basicLoss = self.basicLossFunc(inputImg*weight, target*weight)
        return basicLoss

    
if __name__ == "__main__":
    loss = ColorLoss()
    cosEbLoss = torch.nn.CosineEmbeddingLoss(reduction = "mean")
    
    x = torch.from_numpy(np.array([[[[1, 1], [1, 1]],
                                   [[1, 1], [1, 1]],
                                   [[1, 1], [1, 1]]]]).astype(np.float32))
    
    y = torch.from_numpy(np.array([[[[2, 1], [1, 1]],
                                   [[2, 1], [1, 1]],
                                   [[2, 1], [1, 1]]]]).astype(np.float32))
    
    print(loss(x, y))
    # print(x.shape)
    # print(loss(x, y))