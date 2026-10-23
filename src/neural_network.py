import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights,\
    resnet50, ResNet50_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights, vit_b_16, ViT_B_16_Weights, \
    densenet201, DenseNet201_Weights, \
    Wide_ResNet101_2_Weights, wide_resnet101_2
#from torchsummary import summary
#from captum.attr import IntegratedGradients
from config_loader import FeatureConfig
from torchvision.models import resnet18, ResNet18_Weights
import copy
    
class GixdModel(nn.Module):
    def __init__(self, num_phases=1, use_gmm=False, num_gaussians=None):
        super().__init__()
        #self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.feature_config = FeatureConfig()
        self.num_phases = num_phases
        self.use_gmm = use_gmm
        self.num_gaussians = num_gaussians
        if not self.use_gmm:
            self.output = nn.Sequential(
                nn.Linear(1000,256),
                nn.SiLU(),
                nn.Linear(256,64),
                nn.SiLU(),
                nn.Linear(64,6*num_phases)
                )
        else:
            self.output = nn.Sequential(
                nn.Linear(1000,512),
                nn.SiLU(),
                nn.Linear(512, self.num_gaussians * (6*2+1))
            )
        
    def forward(self, x, top_layer_idx=None):
        x = self.backbone(x)
        x = self.output(x)

        if self.use_gmm:
            x = x.view(-1, self.num_gaussians, 6*2+1)
            means = x[:,:,:6]
            log_vars = x[:,:,6:2*6]
            weights = x[:,:,-1]
            vars = torch.exp(log_vars)
            weights = torch.softmax(weights, dim=-1)
            return means, vars, weights

        return x

class GixdPointModel(nn.Module):
    """
    A point estimator for GIXD regression with two interchangeable modes:
      • multi_head: separate head for each top_layer_index (50 heads)
      • single_head: one shared head + embedding for top_layer_index

    Args:
        num_phases (int): number of phases (6 outputs per phase)
        multi_head (bool): if True use 50 independent heads; if False use single head + embedding
        embed_dim (int): embedding dimension (only used if multi_head=False)
    """
    def __init__(self, num_phases: int, use_transformer = False,
                 multi_head: bool = False, embed_dim: int = 16,
                 pretrained: bool = True):
        super().__init__()
        self.num_phases = num_phases
        self.multi_head = multi_head
        self.feature_config = FeatureConfig()
        self.use_transformer = use_transformer
        
        if self.use_transformer:
          self.transformer = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
          self.transformer.heads.head = nn.Sequential(
              nn.Linear(self.transformer.heads.head.in_features, 256),
              nn.SiLU(),
              nn.Linear(256, 256)
          )
          feat_dim = 256
          self.backbone = self.transformer
        else:
          backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
          # Find feature dimension from final Linear layer
          feat_dim = None
          for layer in backbone.classifier:
              if isinstance(layer, nn.Linear):
                  feat_dim = layer.in_features
                  break
          assert feat_dim is not None, "Could not find Linear layer in backbone.classifier"
          backbone.classifier = nn.Identity()
          self.backbone = backbone.to(memory_format=torch.channels_last)

        out_dim = 6 * num_phases
        num_heads = len(self.feature_config.miller_indices)

        if multi_head:
            # Build one identical head per orientation
            head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, 64),
                nn.SiLU(),
                nn.Linear(64, out_dim)
            )
            self.heads = nn.ModuleList([copy.deepcopy(head) for _ in range(num_heads)])
        else:
            # Shared head + embedding
            self.embedding = nn.Embedding(num_heads, embed_dim)
            self.head = nn.Sequential(
                nn.Linear(feat_dim + embed_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, 64),
                nn.SiLU(),
                nn.Linear(64, out_dim)
            )

    def forward_all_orientations(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only [images, orientations, parameters], one backbone pass.

        Linear heads are evaluated with stacked weights; checkpoint names and
        training forward/gradients are unchanged. Floating-point operation
        grouping differs, so parity is numerical rather than bitwise.
        """
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Use eval() and torch.no_grad() for all-orientation inference")
        if x.ndim == 4 and not getattr(self, "use_transformer", False):
            x = x.contiguous(memory_format=torch.channels_last)
        feats = self.backbone(x)
        if not self.multi_head:
            ids = torch.arange(self.embedding.num_embeddings, device=x.device)
            features = feats[:, None].expand(-1, len(ids), -1)
            embeds = self.embedding(ids)[None].expand(len(feats), -1, -1)
            return self.head(torch.cat((features, embeds), -1))
        values = feats[:, None].expand(-1, len(self.heads), -1)
        for layer_index, layer in enumerate(self.heads[0]):
            if isinstance(layer, nn.Linear):
                weights = torch.stack([head[layer_index].weight for head in self.heads])
                biases = torch.stack([head[layer_index].bias for head in self.heads])
                values = torch.einsum("bhi,hoi->bho", values, weights) + biases
            elif isinstance(layer, nn.SiLU):
                values = F.silu(values)
            else:
                raise TypeError(f"Unsupported all-orientation head layer: {type(layer)}")
        return values

    def forward(self, x: torch.Tensor, top_layer_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass accepts batch of images and corresponding top_layer indices"""
        if x.ndim == 4 and not self.use_transformer:
            # NHWC simulator images already acquire this physical layout when
            # permuted to NCHW. This is a no-op for those inputs and handles
            # other callers without changing logical tensor dimensions.
            x = x.contiguous(memory_format=torch.channels_last)
        feats = self.backbone(x)
        top_layer_idx = top_layer_idx.view(-1).long()

        if self.multi_head:
            # Run each active orientation head on its whole sub-batch. Copy
            # routing metadata once, instead of synchronizing via .item() for
            # every sample; retain head modules and checkpoint parameter names.
            order = torch.argsort(top_layer_idx, stable=True)
            sorted_idx = top_layer_idx[order]
            head_ids, counts = torch.unique_consecutive(sorted_idx, return_counts=True)
            routing = torch.stack((head_ids, counts)).detach().cpu().tolist()
            sorted_feats = feats[order]
            outputs = []
            start = 0
            for head_id, count in zip(*routing):
                outputs.append(self.heads[head_id](sorted_feats[start:start + count]))
                start += count
            return torch.cat(outputs, dim=0)[torch.argsort(order)]
        else:
            embed = self.embedding(top_layer_idx)
            combined = torch.cat([feats, embed], dim=1)
            return self.head(combined)

