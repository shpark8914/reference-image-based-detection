import torch
import glob
from torchvision import transforms
import PIL
import os

class PCB_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, Train=True):
        super().__init__()

        self.data_path = data_path
        self.Train = Train
        
        # Define paths for Normal and Abnormal folders
        self.class_folders = {
            "Normal": os.path.join(data_path, "Normal"),
            "Abnormal": os.path.join(data_path, "Abnormal")
        }
        
        self.data = []
        class_name_set = []

        # Load images from both Normal and Abnormal folders
        for class_name, folder_path in self.class_folders.items():
            class_name_set.append(class_name)
            for img_path in glob.glob(os.path.join(folder_path, "*Current.jpg")):
                # Append both Current image path and its corresponding class (Normal/Abnormal)
                self.data.append([img_path, class_name])

        # Sort class names alphabetically and create class-to-index mapping
        class_name_set.sort()
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_name_set)}

        # Define transformations for training and validation/testing
        self.Train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.Valid_Test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # Get the current image path and its class (Normal/Abnormal)
        Current_img_path, class_name = self.data[idx]

        # Construct the path to the corresponding Master image
        Master_img_path = Current_img_path.replace("Current", "Master")

        # Load images and apply transformations
        if self.Train:
            Current_img = self.Train_transform(PIL.Image.open(Current_img_path))
            Master_img = self.Train_transform(PIL.Image.open(Master_img_path))
        else:
            Current_img = self.Valid_Test_transform(PIL.Image.open(Current_img_path))
            Master_img = self.Valid_Test_transform(PIL.Image.open(Master_img_path))

        # Get label index from the class name (Normal/Abnormal)
        label = self.class_to_idx[class_name]

        return Current_img, Master_img, Current_img_path, label

