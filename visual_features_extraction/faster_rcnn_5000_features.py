
import torchvision.models as models
import copy
import torch.nn as nn
import torch
from collections import OrderedDict
import os
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
import pandas as pd
import tensorflow as tf
from PIL import Image

class vis_Features_model(nn.Module):

    def __init__(self,frcnn_model):
        super().__init__()
        self.frcnn = frcnn_model
        self.transform = copy.deepcopy(frcnn_model.transform)
        self.backbone = copy.deepcopy(frcnn_model.backbone)
        self.rpn = copy.deepcopy(frcnn_model.rpn)
        self.box_roi_pool = copy.deepcopy(frcnn_model.roi_heads.box_roi_pool)
    
    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        return detections

    
    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor(tuples)]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)


        if isinstance(features, torch.Tensor):  ## feature = feature maps
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        box_features = self.box_roi_pool(features, proposals, images.image_sizes)
        return box_features

      
 if __name__ == "__main__":
      fast_rcnn_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True,box_score_thresh=0.001)
      model=vis_Features_model(fast_rcnn_model)
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      model.to(device)
      path='/content/drive/MyDrive/folders'
      folders=os.listdir(path)
      ex_folders=[]
      iter=0
      df=pd.DataFrame()
      model.eval()
      for i in range(len(folders)):
        path='/content/drive/MyDrive/folders/'+str(i+1)+'/news_img.jpg' 
        img = Image.open(path)
        img = img.resize((224,224), Image.ANTIALIAS)
        img = img.convert('RGB')
        x = image.img_to_array(img)
        #x = np.expand_dims(x, axis=0) 
        x = preprocess_input(x) 
        x = torch.from_numpy(x.copy())
        #x=x.permute(0,3,1,2)
        x=(x.permute(2,0,1)).to(device)
        features = model.forward([x]) 
        features_reduce = features.squeeze()
        features_reduce=features_reduce[:5,:40,:5,:5]
