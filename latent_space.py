from re import T
import torch
from torch.nn.functional import normalize

def hac_sl_ratio_loss(distance_fn, vectors, token_mask,y):
    """
    distance_fn: function that takes two tensors of shape (bs,tokens,latent_space_size) and returns a tensor of shape (bs,tokens,tokens)
    vectors: tensor of shape (bs,tokens,latent_space_size)
    token_mask: tensor of shape (bs,tokens)
    y: tensor of shape (bs,num_clusters,tokens)
    """
    is_cluster=torch.any(y==1,dim=-1)#(bs,num_clusters)
    count_clusters=torch.sum(is_cluster,dim=1)#(bs,)
    distances=distance_fn(vectors,vectors)#(bs,tokens,tokens)
    masks=token_mask.unsqueeze(1).float()#(bs,1,tokens)
    mask_mat=torch.matmul(masks.transpose(-2,-1),masks)#(bs,tokens,tokens)
    distances=distances*mask_mat#zeros where mask is 0, distance otherwise
    orig_shape=y.shape
    num_clusters=orig_shape[1]
    if num_clusters==0:
        return None,distances
    y=y.reshape(-1,y.shape[2]).float()#(bs*num_clusters,tokens)
    y=y.unsqueeze(1)#(bs*num_clusters,1,tokens)
    y_t=y.transpose(-2,-1)#(bs*num_clusters,tokens,1)
    expanded_mask_mat=mask_mat.unsqueeze(1).expand(orig_shape[0],num_clusters,orig_shape[2],orig_shape[2])#(bs,num_clusters,tokens,tokens)
    ic_mask=torch.matmul(y_t,y)#(bs*num_clusters,tokens,tokens)
    ic_mask=ic_mask.reshape(orig_shape[0],orig_shape[1],orig_shape[2],orig_shape[2])#(bs,num_clusters,tokens,tokens)
    y_c=torch.abs(y-1)#0s are 1s and 1s are 0s (bs*num_clusters,tokens,tokens)
    ec_mask=torch.matmul(y_t,y_c)#(bs*num_clusters,tokens,tokens)
    ec_mask=ec_mask.reshape(orig_shape[0],orig_shape[1],orig_shape[2],orig_shape[2])*expanded_mask_mat#(bs,num_clusters,tokens,tokens)
    expanded_distances=distances.unsqueeze(1).expand(orig_shape[0],num_clusters,orig_shape[2],orig_shape[2])#(bs,num_clusters,tokens,tokens)
    ic_distances=expanded_distances*ic_mask#(bs,num_clusters,tokens,tokens)
    max_ic_distances=torch.max(ic_distances.reshape(orig_shape[0],num_clusters,-1),dim=-1)[0]#(bs,num_clusters)
    max_distance=torch.max(expanded_distances).detach()
    max_distance=max_distance.expand_as(expanded_distances)#(bs,num_clusters,tokens,tokens) TODO optimize
    ec_distances=torch.where(ec_mask==1,expanded_distances,max_distance)
    min_ec_distances=torch.min(ec_distances.reshape(orig_shape[0],num_clusters,-1),dim=-1)[0]#(bs,num_clusters)
    ratio=(max_ic_distances+1)/(min_ec_distances+1)#(bs,num_clusters)
    ratio_mask=is_cluster
    sentence_loss=torch.sum(ratio*ratio_mask,dim=1)/torch.max(count_clusters,torch.ones_like(count_clusters))#(bs,)
    if torch.all(torch.isnan(sentence_loss)):
        return None,distances
    loss=torch.nanmean(sentence_loss)
    del masks
    del mask_mat
    del ic_mask
    del ec_mask
    del expanded_mask_mat
    #torch.cuda.empty_cache()
    return loss,distances

def hac_sl_ratio_loss_token_based(distance_fn, vectors, token_mask,y):
    """
    computes the ratio for each token in the cluster and then averages the ratios for each cluster
    distance_fn: function that takes two tensors of shape (bs,tokens,latent_space_size) and returns a tensor of shape (bs,tokens,tokens)
    vectors: tensor of shape (bs,tokens,latent_space_size)
    token_mask: tensor of shape (bs,tokens)
    y: tensor of shape (bs,num_clusters,tokens)
    """
    is_cluster=torch.any(y==1,dim=-1)#(bs,num_clusters)
    count_clusters=torch.sum(is_cluster,dim=1)#(bs,)
    distances=distance_fn(vectors,vectors)#(bs,tokens,tokens)
    masks=token_mask.unsqueeze(1).float()#(bs,1,tokens)
    mask_mat=torch.matmul(masks.transpose(-2,-1),masks)#(bs,tokens,tokens)
    distances=distances*mask_mat#zeros where mask is 0, distance otherwise
    orig_shape=y.shape
    num_clusters=orig_shape[1]
    if num_clusters==0:
        return None,distances
    y=y.reshape(-1,y.shape[2]).float()#(bs*num_clusters,tokens)
    y=y.unsqueeze(1)#(bs*num_clusters,1,tokens)
    y_t=y.transpose(-2,-1)#(bs*num_clusters,tokens,1)
    expanded_mask_mat=mask_mat.unsqueeze(1).expand(orig_shape[0],num_clusters,orig_shape[2],orig_shape[2])#(bs,num_clusters,tokens,tokens)
    ic_mask=torch.matmul(y_t,y)#(bs*num_clusters,tokens,tokens)
    ic_mask=ic_mask.reshape(orig_shape[0],orig_shape[1],orig_shape[2],orig_shape[2])#(bs,num_clusters,tokens,tokens)
    y_c=torch.abs(y-1)#0s are 1s and 1s are 0s (bs*num_clusters,tokens,tokens)
    ec_mask=torch.matmul(y_t,y_c)#(bs*num_clusters,tokens,tokens)
    ec_mask=ec_mask.reshape(orig_shape[0],orig_shape[1],orig_shape[2],orig_shape[2])*expanded_mask_mat#(bs,num_clusters,tokens,tokens)
    expanded_distances=distances.unsqueeze(1).expand(orig_shape[0],num_clusters,orig_shape[2],orig_shape[2])#(bs,num_clusters,tokens,tokens)
    ic_distances=expanded_distances*ic_mask#(bs,num_clusters,tokens,tokens)
    #max_ic_distances=torch.max(ic_distances.reshape(orig_shape[0],num_clusters,-1),dim=-1)[0]#(bs,num_clusters)
    max_ic_distances=torch.max(ic_distances,dim=-1)[0]#(bs,num_clusters,tokens)
    max_distance=torch.max(expanded_distances).detach()
    max_distance=max_distance.expand_as(expanded_distances)#(bs,num_clusters,tokens,tokens) TODO optimize
    ec_distances=torch.where(ec_mask==1,expanded_distances,max_distance)
    min_ec_distances=torch.min(ec_distances,dim=-1)[0]#(bs,num_clusters,tokens)
    ratio_token=torch.div((max_ic_distances+1),(min_ec_distances+1))#(bs,num_clusters,tokens)
    ratio=torch.mean(ratio_token,dim=-1)#(bs,num_clusters)
    ratio_mask=is_cluster
    sentence_loss=torch.sum(ratio*ratio_mask,dim=1)/torch.max(count_clusters,torch.ones_like(count_clusters))#(bs,)
    if torch.all(torch.isnan(sentence_loss)):
        return None,distances
    loss=torch.nanmean(sentence_loss)
    del masks
    del mask_mat
    del ic_mask
    del ec_mask
    del expanded_mask_mat
    return loss,distances

def cosine_distance(x,y):
    """
    cosine distance between two tensors
    """
    x_norm=normalize(x,dim=-1)
    y_norm=normalize(y,dim=-1)
    return 1-torch.matmul(x_norm,y_norm.transpose(-2,-1))