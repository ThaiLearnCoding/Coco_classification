import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
from zipfile import ZipFile
from io import BytesIO
from torch.utils.data import Subset


class CLIPGradCAM:
    def __init__(self, model, target_layer_name='ln_post'):
        """
        Khởi tạo Grad-CAM cho CLIP (Vision Transformer).
        model: CLIPFewShotModel (wrapper của bạn).
        target_layer_name: Tên layer cuối cùng trong Vision Transformer của CLIP.
        """
        self.model = model
        self.target_layer = dict([*model.clip_model.visual.named_modules()])[target_layer_name]
        
        self.gradients = None
        self.activations = None
        
        # Đăng ký hook để lấy activations và gradients
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, image_tensor, class_idx):
        self.model.zero_grad()
        
        # Chạy forward pass qua lớp Linear để ra logit phân loại
        logits = self.model(image_tensor)
        
        # Tính gradient cho lớp class_idx
        loss = logits[0, class_idx]
        loss.backward()
        
        # Trích xuất đặc trưng của Vision Transformer (ViT)
        # activations shape: [Sequence_Length, Batch_Size, Transformer_Dim]
        # gradients shape: [Sequence_Length, Batch_Size, Transformer_Dim]
        
        act = self.activations.detach()
        grad = self.gradients.detach()
        
        # Với ViT, token đầu tiên [CLS] chứa thông tin phân loại chung
        # Chúng ta dùng gradient của [CLS] làm trọng số
        weights = grad[0, 0, :].cpu().numpy() # [Transformer_Dim]
        
        # Feature maps là các token còn lại (trừ [CLS])
        # feature_maps shape: [Batch_Size, Tokens_Count, Transformer_Dim]
        feature_maps = act[1:, 0, :].cpu().numpy() 
        
        # Nhân đặc trưng với trọng số
        heatmap = np.zeros(feature_maps.shape[0], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * feature_maps[:, i]
            
        # Reshape heatmap về kích thước lưới ảnh (ví dụ: ViT-B/16@224px là 14x14)
        num_patches = feature_maps.shape[0]
        grid_size = int(np.sqrt(num_patches)) # ví dụ 14 hoặc 21 (với size 336x336)
        heatmap = heatmap.reshape(grid_size, grid_size)
        
        # Chuẩn hóa Heatmap
        heatmap = np.maximum(heatmap, 0) # ReLU
        heatmap /= (np.max(heatmap) + 1e-8)
        
        return heatmap

def plot_grad_cam(original_image, heatmap, target_size=(336, 336)):
    """
    Chồng Heatmap lên ảnh gốc để hiển thị.
    original_image: PIL Image gốc (chưa transform).
    """
    heatmap = cv2.resize(heatmap, target_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    img_np = np.array(original_image.resize(target_size))
    superimposed_img = heatmap * 0.4 + img_np * 0.6
    
    plt.imshow(superimposed_img.astype(np.uint8))
    plt.axis('off')
    plt.show()




def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_training_history(history):
    """Vẽ biểu đồ Loss và Accuracy qua các epoch"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.set_title('Model Loss')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.set_title('Model Accuracy')
    ax2.legend()
    
    plt.show()

def plot_sample_predictions(images, labels, preds, class_names, n=5):
    """Hiển thị một vài ảnh kèm nhãn dự đoán"""
    plt.figure(figsize=(15, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        img = images[i].permute(1, 2, 0).cpu().numpy()
        # Denormalize nếu cần
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        color = 'green' if labels[i] == preds[i] else 'red'
        plt.title(f"L: {class_names[labels[i]]}\nP: {class_names[preds[i]]}", color=color)
        plt.axis('off')
    plt.show()

def display_sample(dataset, index, config):
    # Determine the root dataset and the actual index within it
    if isinstance(dataset, Subset):
        root_dataset = dataset.dataset
        original_idx_in_root = dataset.indices[index]
    else:
        root_dataset = dataset
        original_idx_in_root = index

    # Get the sample's metadata from the root dataset
    metadata_item = root_dataset.filtered_data[original_idx_in_root]

    # Load the original image from the zip file using the file_name from metadata
    zip_path = config['data']['zip_path']
    with ZipFile(zip_path, 'r') as z:
        img_data = z.read(f"images/{metadata_item['file_name']}")
        original_image = Image.open(BytesIO(img_data)).convert('RGB')

    # Get the true labels
    true_labels = metadata_item['labels']

    # Display the image and its labels
    plt.figure(figsize=(7, 7))
    plt.imshow(original_image)
    plt.title(f"Sample Index: {index} (Root Index: {original_idx_in_root})\nTrue Labels: {', '.join(true_labels)}")
    plt.axis('off')
    plt.show()
