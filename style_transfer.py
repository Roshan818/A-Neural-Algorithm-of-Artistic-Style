import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = nn.MSELoss()

    def forward(self, input):
        self.loss_value = self.loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self._gram_matrix(target_feature).detach()
        self.loss = nn.MSELoss()

    def forward(self, input):
        gram = self._gram_matrix(input)
        self.loss_value = self.loss(gram, self.target)
        return input
    
    def _gram_matrix(self, input):
        batch_size, n_channels, h, w = input.size()
        features = input.view(batch_size * n_channels, h * w)
        gram = torch.mm(features, features.t())
        return gram.div(batch_size * n_channels * h * w)

class StyleTransfer:
    def __init__(self, content_img_path, style_img_path, device='cpu', intermediate_callback=None):
        self.device = device
        self.content_img = self._image_loader(content_img_path)
        self.style_img = self._image_loader(style_img_path)
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.model = models.vgg19(pretrained=True).features.to(device).eval()
        self.intermediate_callback = intermediate_callback
        
    def _image_loader(self, image_path):
        loader = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize to fixed size
            transforms.ToTensor()])
        image = Image.open(image_path)
        image = loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)
    
    def _resize_image(self, img):
        return transforms.Resize(self.content_img.shape[2:])(img)
    
    def tensor_to_pil(self, tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)
        return image

    def run_style_transfer(self, num_steps=300, content_weight=1, style_weight=1000000):
        self.style_img = self._resize_image(self.style_img)  # Resize style image
        input_img = self.content_img.clone()
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        
        style_model, style_losses, content_losses = self._get_style_model_and_losses()
        
        run = [0]
        best_loss = float('inf')
        best_image = None
        progress_images = []
        
        while run[0] <= num_steps:
            def closure():
                input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                style_model(input_img)
                
                style_score = 0
                content_score = 0
                
                for sl in style_losses:
                    style_score += sl.loss_value
                for cl in content_losses:
                    content_score += cl.loss_value
                    
                style_score *= style_weight
                content_score *= content_weight
                
                loss = style_score + content_score
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_image = input_img.clone()
                
                loss.backward()
                
                if run[0] % 50 == 0 and self.intermediate_callback:
                    current_img = self.tensor_to_pil(input_img.clone())
                    progress_images.append({
                        'step': run[0],
                        'image': current_img,
                        'loss': loss.item(),
                        'style_loss': style_score.item(),
                        'content_loss': content_score.item()
                    })
                    self.intermediate_callback(progress_images[-1])
                
                run[0] += 1
                return loss
            
            optimizer.step(closure)
        
        input_img.data.clamp_(0, 1)
        return input_img, progress_images, best_image

    def _get_style_model_and_losses(self):
        cnn = copy.deepcopy(self.model)
        content_losses = []
        style_losses = []
        
        model = nn.Sequential()
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
                
            model.add_module(name, layer)
            
            if name in self.content_layers:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)
                
            if name in self.style_layers:
                target_feature = model(self.style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)
            
            # Break the loop if we've added all necessary losses
            if len(content_losses) == len(self.content_layers) and len(style_losses) == len(self.style_layers):
                break
                
        return model, style_losses, content_losses
