import torch
import torch.nn as nn
import torchvision.models as models
import timm
import os

def get_model(model_name='efficientnet_b3', num_classes=2, pretrained=True, weights_path=None):
    """
    Args:
        model_name: 模型名称，如 'resnet50', 'efficientnet_b3'
        num_classes: 分类数
        pretrained: 是否使用预训练权重（仅当 weights_path 为 None 时有效）
        weights_path: 本地权重文件路径，如果指定则忽略 pretrained 参数
    """
    if weights_path is not None:
        pretrained = False

    # 创建模型
    if model_name.startswith('resnet') or model_name.startswith('densenet'):
        # torchvision 模型
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported torchvision model: {model_name}")

        # 替换分类头
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'classifier'):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    else:
        # 使用 timm 创建模型（如 efficientnet）
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    # # 加载本地权重（仅主干部分）
    # if weights_path is not None and os.path.isfile(weights_path):
    #     print(f"Loading weights from {weights_path}...")
    #     # 根据扩展名加载
    #     if weights_path.endswith('.safetensors'):
    #         try:
    #             from safetensors.torch import load_file
    #             state_dict = load_file(weights_path)
    #         except ImportError:
    #             raise ImportError("Please install safetensors: pip install safetensors")
    #     else:
    #         # 对于 .pth 等格式，允许加载旧格式（需设置 weights_only=False）
    #         state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)

    #     # 过滤掉分类头相关的参数（因为我们的模型分类头是 num_classes 类）
    #     # 不同模型分类头命名可能不同，这里针对 timm 的 EfficientNet 使用 'classifier.'
    #     # 对 torchvision 模型可能是 'fc.' 或 'classifier.'，为通用，过滤常见前缀
    #     filtered_state_dict = {}
    #     skipped_keys = []
    #     for k, v in state_dict.items():
    #         # 跳过分类头
    #         if k.startswith('classifier.') or k.startswith('fc.'):
    #             skipped_keys.append(k)
    #             continue
    #         filtered_state_dict[k] = v

    #     if skipped_keys:
    #         print(f"Skipped {len(skipped_keys)} classifier keys: {skipped_keys[:5]}...")  # 只显示前5个

    #     # 加载过滤后的权重，strict=False 允许缺失分类头
    #     missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    #     if missing:
    #         print(f"Missing keys in state_dict (expected): {missing[:5]}...")
    #     if unexpected:
    #         print(f"Unexpected keys in state_dict: {unexpected[:5]}...")
    #     print("Backbone weights loaded successfully.")
    # elif weights_path is not None:
    #     raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # 加载本地权重（全部）
    # model.load_state_dict(torch.load(weights_path))   # 不经过任何过滤

    return model