from re import T
import torch
from torch.nn.functional import normalize

def hac_sl_ratio_loss(distance_fn, vectors, token_mask,y):
    """
    distance_fn: function that takes two tensors of shape (bs,tokens,latent_space_size) and returns a tensor of shape (bs,tokens,tokens)
    vectors: tensor of shape (bs,tokens,latent_space_size)
    token_mask: tensor of shape (bs,tokens)
    y: tensor of shape (bs,tokens)
    """
    orig_shape=y.shape
    distances, ic_distances, ec_distances = hac_compute_distances(distance_fn, vectors, token_mask,y)
    max_ic_distances=torch.max(ic_distances.reshape(orig_shape[0],-1),dim=-1)[0]#(bs,)
    min_ec_distances=torch.min(ec_distances.reshape(orig_shape[0],-1),dim=-1)[0]#(bs,)
    ratio=(max_ic_distances+1)/(min_ec_distances+1)#(bs,)
    batch_loss=ratio
    if torch.all(torch.isnan(batch_loss)):
        return None,distances
    loss=torch.nanmean(batch_loss)
    return loss,distances

def hac_sl_ratio_loss_token_based(distance_fn, vectors, token_mask,y):
    """
    computes the ratio for each token in the cluster and then averages the ratios for each cluster
    distance_fn: function that takes two tensors of shape (bs,tokens,latent_space_size) and returns a tensor of shape (bs,tokens,tokens)
    vectors: tensor of shape (bs,tokens,latent_space_size)
    token_mask: tensor of shape (bs,tokens)
    y: tensor of shape (bs,tokens)
    """
    orig_shape=y.shape
    distances, ic_distances, ec_distances = hac_compute_distances(distance_fn, vectors, token_mask,y)
    max_ic_distances=torch.max(ic_distances,dim=-1)[0]#(bc,tokens)
    min_ec_distances=torch.min(ec_distances,dim=-1)[0]#(bc,tokens)
    ratio_token=torch.div((max_ic_distances+1),(min_ec_distances+1))#(bs,num_clusters,tokens)
    ratio=torch.mean(ratio_token,dim=-1)#(bs,num_clusters)
    loss=torch.nanmean(ratio)
    return loss,distances

def hac_compute_distances(distance_fn, vectors, token_mask,clusters):
    is_cluster=torch.any(clusters==1,dim=-1)#(bs)
    count_clusters=torch.sum(is_cluster)#(bs,)
    distances=distance_fn(vectors,vectors)#(bs,tokens,tokens)
    masks=token_mask.unsqueeze(1).float()#(bs,1,tokens)
    mask_mat=torch.matmul(masks.transpose(-2,-1),masks)#(bs,tokens,tokens)
    distances=distances*mask_mat#zeros where mask is 0, distance otherwise
    orig_shape=clusters.shape
    if count_clusters==0:
        return None,distances
    f=clusters.float() # (bs, tokens)
    f=f.unsqueeze(1)#(bs,tokens,1)
    f_t=f.transpose(-2,-1)#(bs,1,tokens)
    ic_mask=torch.matmul(f_t,f)#(bs,tokens,tokens)
    f_c=torch.abs(f-1)#0s are 1s and 1s are 0s (bs,tokens,1)
    ec_mask=torch.matmul(f_t,f_c)#(bs,tokens,tokens)
    ec_mask=ec_mask*mask_mat#(bs,tokens,tokens)
    ic_distances=distances*ic_mask#(bs,tokens,tokens)
    max_distance=torch.max(distances).detach()#(bs,)
    ec_distances=torch.where(ec_mask==1,distances,max_distance)#(bs,tokens,tokens)
    return distances,ic_distances,ec_distances

def cosine_distance(x,y):
    """
    cosine distance between two tensors
    """
    x_norm=normalize(x,dim=-1)
    y_norm=normalize(y,dim=-1)
    return 1-torch.matmul(x_norm,y_norm.transpose(-2,-1))