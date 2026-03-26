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
        self.model.eval() 
        self.model.zero_grad() 

        # Crucial for Grad-CAM: Ensure the input tensor requires gradients
        # This enables gradient computation even if parts of the model are frozen.
        if not image_tensor.requires_grad:
            image_tensor.requires_grad_(True)

        with torch.enable_grad(): 
            logits = self.model(image_tensor)
            loss = logits[:, class_idx].sum()
            loss.backward()

        # Check if gradients and activations were successfully captured by the hooks
        if self.gradients is None:
            raise RuntimeError("Gradients were not captured. Ensure the target layer is correct and gradients are flowing. "
                            "Also, check if the model's output depends on the target layer in a way that allows gradient flow.")
        if self.activations is None:
            raise RuntimeError("Activations were not captured. Ensure the target layer is correct.")

        act = self.activations.detach()
        grad = self.gradients.detach()

        # The original indexing suggests act/grad are of shape (num_tokens, batch_size, embed_dim)
        weights = grad[0, 0, :].cpu().numpy() # Shape (embed_dim,)

        # Feature maps from other tokens (excluding CLS)
        feature_maps = act[1:, 0, :].cpu().numpy() # Shape (num_patches, embed_dim)

        # Calculate heatmap using dot product of feature maps and weights.
        heatmap = np.dot(feature_maps, weights) # Shape (num_patches,)

        # Reshape the 1D heatmap into a 2D grid corresponding to the patch layout.
        num_patches = feature_maps.shape[0]
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Number of patches ({num_patches}) is not a perfect square. Cannot reshape to square grid for visualization.")
        heatmap = heatmap.reshape(grid_size, grid_size)

        # Apply ReLU to the heatmap and normalize to [0, 1].
        heatmap = np.maximum(heatmap, 0) # Apply ReLU
        heatmap /= (np.max(heatmap) + 1e-8) # Normalize to [0, 1]

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
