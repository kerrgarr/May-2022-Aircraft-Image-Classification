from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import torchvision.transforms.functional as TR
import torchvision.transforms as T

#FAMILY_NAMES = ['A300','A310','A320','A330','A340','A380','ATR-42','ATR-72','An-12','BAE 146','BAE-125','Beechcraft 1900','Boeing 707','Boeing 717','Boeing 727','Boeing 737','Boeing 747','Boeing 757','Boeing 767','Boeing 777','C-130','C-47','CRJ-200','CRJ-700','Cessna 172','Cessna 208','Cessna Citation','Challenger 600','DC-10','DC-3','DC-6','DC-8','DC-9','DH-82','DHC-1','DHC-6','DR-400','Dash 8','Dornier 328','EMB-120','Embraer E-Jet','Embraer ERJ 145','Embraer Legacy 600','Eurofighter Typhoon','F-16','F/A-18','Falcon 2000','Falcon 900','Fokker 100','Fokker 50','Fokker 70','Global Express','Gulfstream','Hawk T1','Il-76','King Air','L-1011','MD-11','MD-80','MD-90','Metroliner','PA-28','SR-20','Saab 2000','Saab 340','Spitfire','Tornado','Tu-134','Tu-154','Yak-42']
FAMILY_NAMES = ['Cessna 172','BAE-125','DR-400','Eurofighter Typhoon','Boeing 747','SR-20']
#FAMILY_NAMES = ['A300','A310','A320','A330','A340','A380']

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

class AircraftDataset(Dataset):
    def __init__(self, dataset_path):
        """
        DataLoader
        """
        import csv
        from os import path
        self.data = []
        transform_image = T.Compose([T.Resize(size=128),T.ToTensor()])
        self.datapath = dataset_path
        with open(path.join(dataset_path, 'labels.csv'), newline='') as f:
            reader = csv.reader(f)
            for fname, label, _ in reader:
                if label in FAMILY_NAMES:
                    image = Image.open(path.join(dataset_path, fname + ".jpg"))
                    label_id = FAMILY_NAMES.index(label)
                    image_resized = transform_image(image)
                    self.data.append((image_resized, label_id))

             
                    
    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]
    

def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset =  AircraftDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


