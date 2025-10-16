# [<title="多模态偏振-RGB融合与动态形态学滤波的透明材料边缘检测">]
# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import glob
import logging
from PIL import Image
import torch.optim as optim

# --- 配置 ---
DATASET_PATH = r"..\\RGBP-Glass\\测试用小样本集\\train"  #自己的训练集路径
OUTPUT_DIR = r"..\\RGBP-Glass\\测试用小样本集"           #输出目录
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.txt") #日志文件路径
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth") #模型保存路径
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_DIR, "output_images") #预测结果保存路径

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# 设置日志
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. 数据集读取与预处理 ---
class GlassEdgeDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir (string): 包含 train 子目录的根目录路径。
            transform (callable, optional): 可选的变换操作。
            is_train (bool): 是否为训练集（需要加载 edge）。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        # 定义子目录路径 - 使用 os.path.join 确保正确性
        self.image_dir = os.path.join(self.root_dir, "image")
        self.aolp_dir = os.path.join(self.root_dir, "aolp")
        self.dolp_dir = os.path.join(self.root_dir, "dolp")
        self.mask_dir = os.path.join(self.root_dir, "mask")
        self.edge_dir = os.path.join(self.root_dir, "edge") if self.is_train else None 

        # 检查必要的目录是否存在
        required_dirs = [self.image_dir, self.aolp_dir, self.dolp_dir, self.mask_dir]
        if self.is_train and self.edge_dir: # 确保 edge_dir 存在再检查
            required_dirs.append(self.edge_dir)
        
        for d in required_dirs:
             # 对于 edge_dir, 因为用了 if is_train else None, 需要额外判断
             if d is not None and not os.path.isdir(d):
                 raise FileNotFoundError(f"Required directory not found: {d}")

        # 获取 image 目录下所有 _rgb.tiff 文件
        # 使用 glob 匹配
        rgb_pattern = os.path.join(self.image_dir, "*_rgb.tiff")
        self.rgb_paths = sorted([os.path.normpath(p) for p in glob.glob(rgb_pattern)])

        if not self.rgb_paths:
            error_msg = f"No RGB files found in {self.image_dir} matching pattern '*_rgb.tiff'"
            logger.warning(error_msg)
            print(f"Warning: {error_msg}")
            self.ids = []
        else:
            logger.info(f"Found {len(self.rgb_paths)} RGB files.")
            print(f"Found {len(self.rgb_paths)} RGB files.")

            # 从路径中提取文件名前缀 (例如从 '.../abc_rgb.tiff' 得到 'abc')
            ids_from_filenames = []
            for p in self.rgb_paths:
                basename_with_ext = os.path.basename(p)
                file_id = os.path.splitext(basename_with_ext)[0].replace('_rgb', '')
                ids_from_filenames.append(file_id)
            # 验证所有对应的文件是否存在
            self.ids = []
            problematic_ids = []
            for file_id in ids_from_filenames:
                # 构建所有预期路径
                expected_files = {
                    'aolp': os.path.join(self.aolp_dir, f"{file_id}_aolp.tiff"),
                    'dolp': os.path.join(self.dolp_dir, f"{file_id}_dolp.tiff"),
                    'mask': os.path.join(self.mask_dir, f"{file_id}_mask.png"), # 注意 .png
                }
                if self.is_train:
                     expected_files['edge'] = os.path.join(self.edge_dir, f"{file_id}_edge.png") # 注意 .png
                
                all_exist = True
                for name, path in expected_files.items():
                    # 规范化路径后再检查
                    norm_path = os.path.normpath(path) 
                    expected_files[name] = norm_path # 更新字典里的路径也为规范化的
                    
                    # print(f"Checking existence of {name}: {norm_path}") 
                    
                    if not os.path.isfile(norm_path):
                        if self.is_train or name != 'edge': # 训练时必须都存在，测试时允许 edge 不存在
                            logger.warning(f"Missing {name} file for ID '{file_id}': {norm_path}")
                            print(f"Warning: Missing {name} file for ID '{file_id}': {norm_path}")
                            all_exist = False
                        
                if all_exist:
                    self.ids.append(file_id)
                else:
                    problematic_ids.append(file_id)
            
            logger.info(f"IDs passed cross-checking: {len(self.ids)}. Problematic IDs (skipped): {problematic_ids}")
            print(f"Valid samples after strict cross-checking: {len(self.ids)}")
            if problematic_ids:
                 print(f"Skipped IDs due to missing files: {problematic_ids}")

    def __len__(self):
        return len(self.ids)

    def _load_image_opencv(self, path, flags=cv2.IMREAD_UNCHANGED):
        """使用 OpenCV 加载图像，支持 .tiff"""
        try:
            # 尝试使用 numpy + imdecode 读取，这对 tiff 更可靠
            img_array = np.fromfile(path, np.uint8)
            if img_array.size == 0:
                 logger.warning(f"Empty file read: {path}")
                 return None
            img = cv2.imdecode(img_array, flags)
            if img is None:
                # Fallback 如果上面的方法不行
                logger.warning(f"imdecode failed for {path}, falling back to imread...")
                img = cv2.imread(path, flags)
            return img
        except Exception as e:
            logger.error(f"Exception while loading image with OpenCV: {path} - {e}")
            return None

    def __getitem__(self, idx):
        if idx >= len(self) or idx < 0:
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}")

        file_id = self.ids[idx]

        # 构建所有文件路径
        rgb_path = os.path.normpath(os.path.join(self.image_dir, f"{file_id}_rgb.tiff"))
        aolp_path = os.path.normpath(os.path.join(self.aolp_dir, f"{file_id}_aolp.tiff"))
        dolp_path = os.path.normpath(os.path.join(self.dolp_dir, f"{file_id}_dolp.tiff"))
        mask_path = os.path.normpath(os.path.join(self.mask_dir, f"{file_id}_mask.png")) # .png
        edge_path = os.path.normpath(os.path.join(self.edge_dir, f"{file_id}_edge.png")) if self.is_train and self.edge_dir else None

        try:
            # --- 加载图像 ---
            # RGB 图像 (3通道)
            rgb_img = self._load_image_opencv(rgb_path, cv2.IMREAD_COLOR)
            if rgb_img is None:
                raise FileNotFoundError(f"Could not load RGB image (or it's corrupted): {rgb_path}")
            # OpenCV 读取为 BGR，转换为 RGB
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            # AoLP 图像 (1通道)
            aolp_img = self._load_image_opencv(aolp_path, cv2.IMREAD_GRAYSCALE)
            if aolp_img is None:
                raise FileNotFoundError(f"Could not load AoLP image (or it's corrupted): {aolp_path}")

            # DoLP 图像 (1通道)
            dolp_img = self._load_image_opencv(dolp_path, cv2.IMREAD_GRAYSCALE)
            if dolp_img is None:
                raise FileNotFoundError(f"Could not load DoLP image (or it's corrupted): {dolp_path}")

            # Mask 图像 (1通道, 二值化)
            mask_img = self._load_image_opencv(mask_path, cv2.IMREAD_GRAYSCALE) # Mask 通常是 .png
            if mask_img is None:
                raise FileNotFoundError(f"Could not load Mask image (check path/format/encoding): {mask_path}. Current working dir: {os.getcwd()}")
            mask_img = (mask_img > 127).astype(np.float32) # 二值化

            # Edge 图像 (1通道, 二值化) - 仅训练时需要
            edge_img = None
            if self.is_train and edge_path:
                edge_img = self._load_image_opencv(edge_path, cv2.IMREAD_GRAYSCALE)
                if edge_img is None:
                    raise FileNotFoundError(f"Could not load Edge image (check path/format/encoding): {edge_path}. Current working dir: {os.getcwd()}")
                edge_img = (edge_img > 127).astype(np.float32) # 二值化

            # --- 应用变换 ---
            if self.transform:
                # 注意：transforms.Compose 通常期望 PIL Image 或 HWC numpy array
                # 但 ToTensor() 期望 HWC。cv2 读取的是 HWC。
                # 如果 transform 包含 ToTensor(), 它会自动处理 HWC -> CHW
                rgb_img = self.transform(rgb_img)
                # 对于单通道图像，ToTensor 也能处理，但输入需要是 (H, W) 或 (H, W, 1) 的 numpy array
                # 我们传入 (H, W) 的 numpy array 即可
                aolp_img = self.transform(aolp_img)
                dolp_img = self.transform(dolp_img)
                mask_img = self.transform(mask_img)
                if self.is_train and edge_img is not None:
                    edge_img = self.transform(edge_img)

            # --- 构建返回项 ---
            item = {
                'rgb': rgb_img,
                'aolp': aolp_img,
                'dolp': dolp_img,
                'mask': mask_img,
                'filename': f"{file_id}" # 可以只存 ID
            }
            if self.is_train and edge_img is not None:
                item['edge'] = edge_img

            return item

        except Exception as e:
            logger.error(f"Error loading data for ID {file_id} (index {idx}): {e}")
            # 重要：在 DataLoader 中，如果 __getitem__ 抛出异常，这个样本会被跳过（如果 num_workers > 0）
            # 或者导致整个训练中断（如果 num_workers = 0）。最好确保数据完整。
            raise e # 让 DataLoader 知道加载失败

# --- 2. 模型定义 ---

# 简单的CNN特征提取器
class SimpleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleFeatureExtractor, self).__init__()
        # 使用较小的卷积核和步长来模拟小波变换的部分效果
        # 或者直接输出特征图，让后续融合模块处理
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# 通道注意力机制 (Squeeze-and-Excitation)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1) # 添加最大池化
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y_avg = self.fc(y_avg)
        y_max = self.fc(y_max)
        y = self.sigmoid(y_avg + y_max).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 融合模块 (生成 Ifused 和 Ffused)
class FusionModule(nn.Module):
    def __init__(self, channels):
        super(FusionModule, self).__init__()
        self.channels = channels
        # 用于加权融合 Ffused 的通道注意力
        self.ca = ChannelAttention(channels * 2) # 输入是拼接后的 F_rgb 和 F_p

    def forward(self, F_rgb, F_p):
        """
        Args:
            F_rgb: Tensor, (B, C, H, W) RGB特征图 (模拟低频)
            F_p: Tensor, (B, C, H, W) 偏振特征图 (AoLP+DoLP concatenate, 模拟高频)
        Returns:
            Ifused: Tensor, (B, C, H, W) 融合图像特征 (偏振高频 + RGB低频)
            F_fused: Tensor, (B, C, H, W) 通道注意力加权融合特征
        """
        # --- 1. 生成 Ifused (模拟小波融合) ---
        # 根据描述：保留小波分解后偏振的高频边缘与 RGB 的低频纹理
        # 这里我们简化为 F_p (高频模拟) + F_rgb (低频模拟)
        # 在实际应用中，应在特征提取前或后进行小波变换，并分别提取对应分量。
        Ifused = F_p + F_rgb # (B, C, H, W)

        # --- 2. 生成 Ffused (通道注意力加权融合) ---
        # 拼接特征
        fused_cat = torch.cat([F_rgb, F_p], dim=1) # (B, 2*C, H, W)
        # 通道注意力加权
        weighted_fused = self.ca(fused_cat) # (B, 2*C, H, W)
        # 分离加权后的特征
        F_rgb_weighted, F_p_weighted = torch.split(weighted_fused, self.channels, dim=1)
        # 加权融合 F_fused = Wc ⊙ F_rgb + (1-Wc) ⊙ F_p
        F_fused = (F_rgb_weighted + F_p_weighted) / 2.0 # (B, C, H, W)

        return Ifused, F_fused

# MLP解码器 (修改为接收 Ifiltered 和 Ffused 输入)
class MLPDecoder(nn.Module):
    def __init__(self, input_channels_ifiltered, input_channels_ffused, mid_channels=64, output_channels=1):
        super(MLPDecoder, self).__init__()
        total_input_channels = input_channels_ifiltered + input_channels_ffused
        self.decoder = nn.Sequential(
            nn.Conv2d(total_input_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, output_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, ifiltered, ffused):
        # 将 Ifiltered 和 Ffused 在通道维度上拼接
        combined_features = torch.cat([ifiltered, ffused], dim=1) # (B, C1+C2, H, W)
        # 解码输出边缘图
        edge_pred = self.decoder(combined_features)
        return edge_pred

# 完整模型 (修改前向传播以包含 Hessian 和 Morphological Filtering)
class GlassEdgeDetector(nn.Module):
    def __init__(self):
        super(GlassEdgeDetector, self).__init__()
        self.feature_channels = 16
        self.rgb_extractor = SimpleFeatureExtractor(3, self.feature_channels)
        self.pol_extractor = SimpleFeatureExtractor(2, self.feature_channels) # AoLP + DoLP
        self.fusion_module = FusionModule(self.feature_channels)
        # 解码器接收 Ifiltered 和 Ffused 的通道数
        self.decoder = MLPDecoder(self.feature_channels, self.feature_channels)

    def forward(self, rgb, aolp, dolp):
        # 1. 特征嵌入
        F_rgb = self.rgb_extractor(rgb)
        pol_input = torch.cat([aolp, dolp], dim=1)
        F_p = self.pol_extractor(pol_input)

        # 2. 融合，得到 Ifused 和 Ffused
        Ifused, F_fused = self.fusion_module(F_rgb, F_p)

        # 3. Hessian 矩阵计算局部曲率 B (简化实现，作用于 Ifused)
        # 注意：Hessian计算通常在CPU/Numpy上进行，这里为了流程完整性，在PyTorch中模拟
        # 实际应用中可能需要更复杂的实现或与CPU代码交互
        B = compute_hessian_curvature_approx(Ifused) # (B, C, H, W)

        # 4. 动态开运算 Ifiltered = (Ifused ∘ B) ∙ B (简化实现)
        # 注意：形态学操作通常在CPU/OpenCV上进行，这里简化为基于曲率的滤波
        Ifiltered = dynamic_morphological_filtering_approx(Ifused, B) # (B, C, H, W)

        # 5. 解码，使用 Ifiltered 和 Ffused 作为输入
        edge_pred = self.decoder(Ifiltered, F_fused)
        return edge_pred

# --- 3. 辅助函数 ---

# 小波变换 (简化版，仅用于特征增强)
# def wavelet_transform(img_tensor):
#     # 此处为简化，实际应用中可以更复杂
#     # 我们在特征提取器中通过CNN学习小波-like特征
#     # 这里直接返回输入，表示CNN内部处理
#     return img_tensor

# Hessian矩阵计算局部曲率 (简化近似版，用于PyTorch张量)
def compute_hessian_curvature_approx(img_tensor):
    """
    简化版本的Hessian曲率计算，适用于PyTorch张量。
    使用二阶差分近似二阶导数。
    """
    # img_tensor: (B, C, H, W)
    batch_size, channels, h, w = img_tensor.shape
    
    # 计算二阶导数 (使用差分近似)
    # Hxx (二阶x导数)
    pad_img = torch.nn.functional.pad(img_tensor, (1, 1, 1, 1), mode='replicate')
    Hxx = pad_img[:, :, 2:, 1:-1] - 2 * pad_img[:, :, 1:-1, 1:-1] + pad_img[:, :, :-2, 1:-1]
    
    # Hyy (二阶y导数)
    Hyy = pad_img[:, :, 1:-1, 2:] - 2 * pad_img[:, :, 1:-1, 1:-1] + pad_img[:, :, 1:-1, :-2]
    
    # Hxy (混合二阶导数)
    Hxy = (pad_img[:, :, 2:, 2:] - pad_img[:, :, 2:, :-2] - pad_img[:, :, :-2, 2:] + pad_img[:, :, :-2, :-2]) / 4.0

    # 计算判别式
    det_H = Hxx * Hyy - Hxy * Hxy
    trace_H = Hxx + Hyy
    # 避免除零，计算曲率 B (这里简化为 (det_H / (trace_H + eps)) 的平方根的绝对值)
    eps = 1e-8
    B = torch.abs(torch.sqrt(torch.abs(det_H / (trace_H + eps)) + eps))
    
    # 确保输出大小与输入一致 (差分可能导致边缘信息丢失)
    # 这里假设填充后大小一致，实际可能需要裁剪或插值
    # 为简化，我们直接返回计算结果
    return B # (B, C, H, W)

# 动态形态学滤波 (简化近似版)
def dynamic_morphological_filtering_approx(input_tensor, structuring_element):
    """
    对输入张量的每个通道应用近似的动态形态学滤波。

    Args:
        input_tensor (torch.Tensor): 输入张量，形状为 (B, C, H, W)。
        structuring_element (torch.Tensor or np.ndarray): 结构元素。

    Returns:
        torch.Tensor: 滤波后的张量，形状为 (B, C, H, W)。
    """
    device = input_tensor.device
    B, C, H, W = input_tensor.shape

    # 1. 基础平滑滤波 - 使用 Depthwise Convolution
    # 创建一个适用于 Depthwise Conv 的平均核
    # --- 关键修改: 正确的 Depthwise Conv 权重形状 ---
    kernel_size = 3
    # 权重形状: (输出通道数=C, 输入通道数每组=1, K, K)
    avg_kernel = torch.ones(C, 1, kernel_size, kernel_size, device=device, dtype=input_tensor.dtype) / (kernel_size * kernel_size)

    base_filtered = F.conv2d(
        input_tensor,       # (B, C, H, W)
        avg_kernel,         # (C, 1, K, K)
        padding=kernel_size // 2,
        groups=C            # Depthwise
    )

    # 2. 计算梯度 (Sobel 算子近似) - 使用 Depthwise Convolution
    # --- 修正 2: Depthwise Conv 权重 for Sobel X ---
    sobel_x_kernel_base = torch.tensor([[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]], dtype=input_tensor.dtype, device=device)
    # 扩展为 (C, 1, 3, 3) 形状以用于 Depthwise Conv
    sobel_x_kernel = sobel_x_kernel_base.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    # --- 修正 3: Depthwise Conv 权重 for Sobel Y ---
    sobel_y_kernel_base = torch.tensor([[-1, -2, -1],
                                        [ 0,  0,  0],
                                        [ 1,  2,  1]], dtype=input_tensor.dtype, device=device)
    # 扩展为 (C, 1, 3, 3) 形状以用于 Depthwise Conv
    sobel_y_kernel = sobel_y_kernel_base.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    grad_x = F.conv2d(
        base_filtered,      # (B, C, H, W)
        sobel_x_kernel,     # (C, 1, 3, 3)
        padding=1,
        groups=C            # Depthwise
    )
    grad_y = F.conv2d(
        base_filtered,      # (B, C, H, W)
        sobel_y_kernel,     # (C, 1, 3, 3)
        padding=1,
        groups=C            # Depthwise
    )
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8) # 防止 sqrt(0)

    # 3. 动态阈值 (简化版)
    # 使用全局平均作为阈值的代理
    threshold = torch.mean(gradient_magnitude, dim=(2, 3), keepdim=True) # (B, C, 1, 1)
    threshold = threshold.expand_as(gradient_magnitude) # (B, C, H, W)

    # 4. 应用阈值
    filtered = torch.where(gradient_magnitude > threshold, base_filtered, input_tensor)

    return filtered

# --- 4. 训练与评估 ---

def dice_coefficient(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice.item()

def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = len(data_loader)
    
    for batch_idx, batch in enumerate(data_loader):
        # --- 关键修改 1: 将数据移到 device (GPU/CPU) ---
        rgb = batch['rgb'].to(device)
        aolp = batch['aolp'].to(device)
        dolp = batch['dolp'].to(device)
        edge_gt = batch['edge'].to(device) # Ground Truth Edge
        mask = batch['mask'].to(device)    # Mask
        
        optimizer.zero_grad()
        
        # 前向传播
        edge_pred = model(rgb, aolp, dolp) # (B, 1, H, W)
        
        # 应用 Mask
        masked_pred = edge_pred * mask
        masked_gt = edge_gt * mask
        
        # 计算损失
        loss = criterion(masked_pred, masked_gt)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 计算 Dice 系数 (通常在阈值化后计算，这里简化)
        with torch.no_grad():
            dice = dice_coefficient(masked_pred, masked_gt)
        
        total_loss += loss.item()
        total_dice += dice
        
        if batch_idx % 2 == 0: # 每 N 个 batch 打印一次
            print(f"Epoch [{epoch+1}], Batch [{batch_idx}/{num_batches}], Loss: {loss.item():.4f}, Dice: {dice:.4f}")
            logger.info(f"Epoch [{epoch+1}], Batch [{batch_idx}/{num_batches}], Loss: {loss.item():.4f}, Dice: {dice:.4f}")

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    print(f"Epoch [{epoch+1}] Average Training Loss: {avg_loss:.4f}, Average Dice: {avg_dice:.4f}")
    logger.info(f"Epoch [{epoch+1}] Average Training Loss: {avg_loss:.4f}, Average Dice: {avg_dice:.4f}")
    return avg_loss, avg_dice

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for batch in data_loader:
            # --- 关键修改 2: 将验证数据也移到 device ---
            rgb = batch['rgb'].to(device)
            aolp = batch['aolp'].to(device)
            dolp = batch['dolp'].to(device)
            edge_gt = batch['edge'].to(device)
            mask = batch['mask'].to(device)
            
            # 前向传播
            edge_pred = model(rgb, aolp, dolp)
            
            # 应用 Mask
            masked_pred = edge_pred * mask
            masked_gt = edge_gt * mask
            
            # 计算损失
            loss = criterion(masked_pred, masked_gt)
            
            # 计算 Dice 系数
            dice = dice_coefficient(masked_pred, masked_gt)
            
            total_loss += loss.item()
            total_dice += dice
            
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    print(f"Evaluation - Average Loss: {avg_loss:.4f}, Average Dice: {avg_dice:.4f}")
    logger.info(f"Evaluation - Average Loss: {avg_loss:.4f}, Average Dice: {avg_dice:.4f}")
    return avg_loss, avg_dice

def save_prediction_images(model, data_loader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    sigmoid = nn.Sigmoid() # 如果模型最后没有 Sigmoid，这里加上
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # --- 关键修改 3: 将预测数据也移到 device ---
            rgb = batch['rgb'].to(device)
            aolp = batch['aolp'].to(device)
            dolp = batch['dolp'].to(device)
            mask = batch['mask'].to(device)
            filenames = batch['filename']
            
            # 前向传播
            edge_logits = model(rgb, aolp, dolp) # 获取 logits
            edge_probs = sigmoid(edge_logits)   # 应用 Sigmoid 得到概率
            
            # 应用 Mask
            masked_probs = edge_probs * mask
            
            # CPU 回传以便保存
            preds_cpu = masked_probs.cpu().numpy() # Shape: (B, 1, H, W)
            
            for j in range(preds_cpu.shape[0]):
                filename_base = filenames[j]
                pred_img = preds_cpu[j, 0] # 取第一个通道 (H, W)
                
                # 转换为 0-255 uint8 图像
                pred_img_uint8 = (pred_img * 255).astype(np.uint8)
                
                # 保存图像
                output_path = os.path.join(output_dir, f"{filename_base}_pred.png")
                success = cv2.imwrite(output_path, pred_img_uint8)
                if success:
                    logger.info(f"Saved prediction: {output_path}")
                else:
                    logger.warning(f"Failed to save prediction: {output_path}")

# --- 新增函数: 预测并保存 (用于测试集) ---
def predict_and_save(model, data_loader, device, output_dir):
    """
    使用训练好的模型对数据集进行预测并将结果保存为图像。
    不需要 ground truth edge。
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    sigmoid = nn.Sigmoid()
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # --- 关键修改: 将数据移到 device ---
            rgb = batch['rgb'].to(device)
            aolp = batch['aolp'].to(device)
            dolp = batch['dolp'].to(device)
            mask = batch['mask'].to(device)
            filenames = batch['filename'] # 文件名列表
            
            # 前向传播得到预测 logits
            edge_logits = model(rgb, aolp, dolp) # (B, 1, H, W)
            
            # 应用 Sigmoid 得到概率
            edge_probs = sigmoid(edge_logits)   # (B, 1, H, W)
            
            # 应用 Mask
            masked_probs = edge_probs * mask    # (B, 1, H, W)
            
            # CPU 回传以便保存
            preds_cpu = masked_probs.cpu().numpy() # Shape: (B, 1, H, W)
            
            for j in range(preds_cpu.shape[0]):
                filename_base = filenames[j]
                pred_img = preds_cpu[j, 0] # 取第一个通道 (H, W)
                
                # 转换为 0-255 uint8 图像
                pred_img_uint8 = (pred_img * 255).astype(np.uint8)
                
                # 保存图像
                output_path = os.path.join(output_dir, f"{filename_base}_predicted_edge.png") # --- 修改文件名后缀 ---
                # success = cv2.imwrite(output_path, pred_img_uint8)
                # if success:
                #     logger.info(f"[Prediction] Saved prediction: {output_path}")
                #     print(f"[Prediction] Saved: {output_path}") # --- 添加控制台输出 ---
                # else:
                #     logger.warning(f"[Prediction] Failed to save prediction: {output_path}")
                #     print(f"[Prediction] Failed to save: {output_path}")
                try:
                    # 使用PIL保存（支持中文路径）
                    from PIL import Image
                    img = Image.fromarray(pred_img_uint8)
                    img.save(output_path)
                    logger.info(f"[Prediction] Saved prediction: {output_path}")
                    print(f"[Prediction] Saved: {output_path}")
                except Exception as e:
                    logger.error(f"[Prediction] Failed to save {output_path}: {str(e)}")
                    print(f"[Prediction] Failed to save: {output_path} - {str(e)}")

def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")

    # 数据变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # --- 训练和验证 ---
    # 创建训练数据集和加载器
    full_dataset = GlassEdgeDataset(root_dir=DATASET_PATH, transform=transform, is_train=True) # 训练集需要 edge
    
    if len(full_dataset) == 0:
        print("No valid training samples found. Exiting.")
        logger.error("No valid training samples found. Exiting.")
        return

    # 简单划分训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    if val_size == 0:
        print("Not enough samples for validation. Using all for training and validation.")
        logger.warning("Not enough samples for validation. Using all for training and validation.")
        train_dataset = full_dataset
        val_dataset = full_dataset # 简单复用
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

    # 初始化模型、损失函数和优化器
    model = GlassEdgeDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 20
    best_val_dice = 0.0

    for epoch in range(num_epochs):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)

        # 保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"New best model saved with Dice: {best_val_dice:.4f}")

    logger.info("Training finished.")
    print("Training finished.")

    # 加载最佳模型进行最终评估和预测保存
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    final_val_loss, final_val_dice = evaluate(model, val_loader, criterion, device)
    logger.info(f"Final Evaluation (Best Model) - Loss: {final_val_loss:.4f}, Dice: {final_val_dice:.4f}")
    print(f"Final Evaluation (Best Model) - Loss: {final_val_loss:.4f}, Dice: {final_val_dice:.4f}")

    # --- 保存验证集的预测结果 ---
    val_output_dir = os.path.join(OUTPUT_IMAGE_DIR, "validation_predictions")
    save_prediction_images(model, val_loader, device, val_output_dir) # 使用原有的函数保存验证集结果
    logger.info("Validation prediction images saved.")
    print("Validation prediction images saved.")


    # --- 对测试集进行预测 ---
    TEST_DATASET_PATH = os.path.join(os.path.dirname(DATASET_PATH), "test") # 例如: "...\\测试用小样本集\\test"
    if os.path.isdir(TEST_DATASET_PATH):
        print(f"\n--- Starting Prediction on Test Set: {TEST_DATASET_PATH} ---")
        logger.info(f"\n--- Starting Prediction on Test Set: {TEST_DATASET_PATH} ---")
        
        # 创建测试数据集和加载器 (注意 is_train=False)
        test_dataset = GlassEdgeDataset(root_dir=TEST_DATASET_PATH, transform=transform, is_train=False) # 不需要 edge
        if len(test_dataset) > 0:
            test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
            
            # 定义测试集预测结果的保存目录
            test_output_dir = os.path.join(OUTPUT_DIR, "test_predictions")
            
            # 调用新的预测函数
            predict_and_save(model, test_loader, device, test_output_dir)
            
            print(f"--- Prediction on Test Set Finished. Results saved to: {test_output_dir} ---")
            logger.info(f"--- Prediction on Test Set Finished. Results saved to: {test_output_dir} ---")
        else:
            print(f"No valid samples found in test dataset: {TEST_DATASET_PATH}")
            logger.warning(f"No valid samples found in test dataset: {TEST_DATASET_PATH}")
    else:
        print(f"Test dataset directory not found: {TEST_DATASET_PATH}. Skipping test prediction.")
        logger.info(f"Test dataset directory not found: {TEST_DATASET_PATH}. Skipping test prediction.")

if __name__ == "__main__":
    main()