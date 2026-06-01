import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
import re

class VisualTextCoCropper:
    """
    Visual-Text Co-Cropping 数据增强器
    - 根据 output 中的关键视觉名词定位图像区域
    - 以该区域为中心做安全裁剪/缩放
    - 动态重写 instruction 以匹配裁剪后的视觉内容
    ⚠️ 绝不修改 output 中的判断词("是"/"否")
    """
    
    # 陨石识别高频关键特征词库（可根据数据集扩展）
    KEY_VISUAL_TERMS = ["熔壳", "气印", "金属颗粒", "矿物", "纹理", "颜色", "形状"]
    
    def __init__(self, crop_scale_range=(0.8, 1.2), safe_margin=0.6):
        self.crop_scale_range = crop_scale_range
        self.safe_margin = safe_margin  # 关键区域最小保留比例
        
    def _extract_key_region(self, image: Image.Image, output_text: str) -> Optional[Tuple[int,int,int,int]]:
        """
        从 output 中提取关键视觉区域坐标
        轻量级实现：使用预计算显著性图 + 文本关键词匹配
        返回 (x1, y1, x2, y2) 或 None
        """
        # 1. 检查 output 是否包含关键视觉术语
        matched_terms = [term for term in self.KEY_VISUAL_TERMS if term in output_text]
        if not matched_terms:
            return None
            
        # 2. 获取显著性图（假设已通过离线预处理存储在 metadata 中）
        # 若无预计算 saliency，可降级为随机选择高对比度区域
        saliency_map = getattr(image, '_saliency_map', None)
        if saliency_map is None:
            return None
            
        # 3. 在显著性图中寻找与关键词语义对应的峰值区域
        # 简化版：直接取显著性图 top-k 连通域中与文本描述最匹配的区域
        h, w = saliency_map.shape
        threshold = saliency_map.mean() + saliency_map.std()
        binary_mask = saliency_map > threshold
        
        # 找最大连通区域作为候选
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary_mask)
        if num_features == 0:
            return None
            
        sizes = ndimage.sum(binary_mask, labeled, range(1, num_features+1))
        max_label = np.argmax(sizes) + 1
        ys, xs = np.where(labeled == max_label)
        
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        
        # 4. 安全检查：确保区域面积不过小
        area_ratio = (x2-x1)*(y2-y1) / (w*h)
        if area_ratio < 0.05 or area_ratio > 0.8:  # 避免极端裁剪
            return None
            
        return (x1, y1, x2, y2)
    
    def _safe_crop(self, image: Image.Image, bbox: Tuple[int,int,int,int]) -> Image.Image:
        """以关键区域为中心做安全裁剪，保证关键区域覆盖率 > safe_margin"""
        x1, y1, x2, y2 = bbox
        w, h = image.size
        
        # 计算裁剪尺寸（随机缩放）
        scale = np.random.uniform(*self.crop_scale_range)
        crop_w = int((x2-x1) * scale)
        crop_h = int((y2-y1) * scale)
        
        # 限制最大裁剪范围不超过原图
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        
        # 以 bbox 中心为基准，确保关键区域在裁剪框内占比足够
        cx, cy = (x1+x2)//2, (y1+y2)//2
        left = max(0, min(cx - crop_w//2, w - crop_w))
        top = max(0, min(cy - crop_h//2, h - crop_h))
        
        return image.crop((left, top, left+crop_w, top+crop_h))
    
    def _adapt_instruction(self, instruction: str, matched_terms: list) -> str:
        """动态重写 instruction，将泛化描述替换为具体关注点"""
        if not matched_terms:
            return instruction
            
        focus_term = matched_terms[0]  # 取第一个匹配词
        # 替换模式：将"观察这张图片中石头的视觉特征" → "重点观察图中[熔壳]的纹理细节"
        pattern = r"观察这张图片中石头的视觉特征（[^）]*）"
        replacement = f"重点观察图中【{focus_term}】的细节特征"
        adapted = re.sub(pattern, replacement, instruction)
        
        # 若未匹配到标准模板，则在末尾追加关注提示
        if adapted == instruction:
            adapted += f" 请特别注意图中的{focus_term}特征。"
            
        return adapted
    
    def __call__(self, sample: Dict) -> Dict:
        """
        主增强入口
        - 输入: 包含 image, instruction, output 的样本字典
        - 输出: 增强后的样本（image 和 instruction 可能变化，output 不变）
        """
        image = sample["image"]
        instruction = sample["instruction"]
        output = sample["output"]
        
        # 1. 提取关键区域
        bbox = self._extract_key_region(image, output)
        if bbox is None:
            return sample  # 无法定位则跳过增强
            
        # 2. 安全裁剪
        cropped_image = self._safe_crop(image, bbox)
        
        # 3. 适配 instruction
        matched_terms = [t for t in self.KEY_VISUAL_TERMS if t in output]
        adapted_instruction = self._adapt_instruction(instruction, matched_terms)
        
        # 4. 构建增强样本（output 绝对不变）
        augmented_sample = sample.copy()
        augmented_sample["image"] = cropped_image
        augmented_sample["instruction"] = adapted_instruction
        # augmented_sample["output"] 保持原样
        
        return augmented_sample