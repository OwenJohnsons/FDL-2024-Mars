import torch
import cv2

# --------------------------------- #
#             Dataset               #
# --------------------------------- #


class ImageDataset(torch.utils.data.Dataset):
        
        def __init__(self,df ,transforms = None):
            self.df = df
            self.transforms = transforms
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self,idx):
            img = cv2.imread(self.df.iloc[idx,0])
            label = self.df.iloc[idx, 1:]
            if self.transforms:
                return self.transforms(torch.Tensor(img).permute([2,1,0])), torch.Tensor([label])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img / 255.
            # AS: Change the dimensions of the image here
            img = cv2.resize(img,(602, 602))
            return torch.Tensor(img).permute([2,1,0]), torch.Tensor(label)