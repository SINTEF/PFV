import numpy as np
import torch
import torch.nn.functional as F

def normalize_and_scale_features(features, n_sigma=1):
    # Whiten
    scaled_features = (features - np.mean(features)) / (np.std(features) )
    # Clip within n standard deviation
    scaled_features = np.clip(scaled_features, -n_sigma, n_sigma)
    # Rescale to [0,1]  **This gives a batch scaling**
    scaled_features = (scaled_features - scaled_features.min()) / (scaled_features.max()-scaled_features.min())
    return scaled_features

def pca_decomposition(x, n_components=3):
    feats = x.permute(0,2,3,1).reshape(-1, x.shape[1])
    feats = feats-torch.mean(feats,0)
    u,s,v = torch.svd(feats, compute_uv=True)
    pc = torch.matmul(u[:,:n_components], torch.diag(s[:n_components]))
    pc = pc.view(x.shape[0], x.shape[2], x.shape[3], 3).permute(0,3,1,2)
    return pc

# Normalized activation maps
def feature_map_normalization(f):
    act_map = torch.sum(f, dim=1).unsqueeze(1)
    act_map /= act_map.max()
    return act_map


# Principle feature visulization
def pfv(embeddings, image_shape=None, idx_layer=None, hierarchical=False, interp_mode='bilinear'):

    if image_shape is None: image_shape = embeddings[0].shape[-2:]
    if idx_layer is None: idx_layer = len(embeddings)

    with torch.no_grad():
        # Decompose to principle contrasting features
        layer_to_visualize = pca_decomposition(embeddings[idx_layer], 3)

        # Weighted upsampling with activation maps
        if hierarchical:
            for f in reversed(embeddings[:idx_layer]):
                layer_to_visualize = F.interpolate(layer_to_visualize, size=(f.shape[2], f.shape[3]), mode=interp_mode)
                layer_to_visualize *= feature_map_normalization(f)
        else:
            amap = [F.interpolate(torch.sum(x,dim=1).unsqueeze(1), size=image_shape,mode=interp_mode) for x in embeddings[:idx_layer]]
            amap = torch.cat(amap, dim=1)
            layer_to_visualize = F.interpolate(layer_to_visualize, size=image_shape,mode=interp_mode) * torch.sum(amap,dim=1).unsqueeze(1)
        
        # Normalize response to RGB
        layer_to_visualize = layer_to_visualize.detach().cpu().numpy()
        rgb = normalize_and_scale_features(layer_to_visualize)
        return rgb

        
            


         




    