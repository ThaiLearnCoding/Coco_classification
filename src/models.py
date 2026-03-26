import torch
import torch.nn as nn
import clip

class CLIPFewShotModel(nn.Module):
    def __init__(self, model_name, num_classes, device):
        super().__init__()
        self.device = device
        # Load CLIP
        self.clip_model, self.preprocess = clip.load(model_name, device=device)
        
        # Đóng băng toàn bộ CLIP để làm Linear Probing (tiết kiệm tài nguyên)
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Thêm Linear Head để phân loại 10 lớp
        input_dim = self.clip_model.visual.output_dim
        self.classifier = nn.Linear(input_dim, num_classes).to(device)

    def forward(self, image):
        # Trích xuất đặc trưng ảnh từ CLIP
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Đưa qua lớp Linear để phân loại
        return self.classifier(image_features.float())