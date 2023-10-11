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
    distances, ic_distances, ec_distances,_,_ = hac_compute_distances(distance_fn, vectors, token_mask,y)
    max_ic_distances=torch.max(ic_distances.reshape(orig_shape[0],-1),dim=-1)[0]#(bs,)
    min_ec_distances=torch.min(ec_distances.reshape(orig_shape[0],-1),dim=-1)[0]#(bs,)
    ratio=(max_ic_distances+1)/(min_ec_distances+1)#(bs,)
    batch_loss=ratio
    if torch.all(torch.isnan(batch_loss)):
        return None,distances
    loss=torch.nanmean(batch_loss)
    return loss,distances

def hac_sl_margin_loss(distance_fn, vectors, token_mask,y, margin=0.01):
    """
    For each cluster member, compute the margin between the max distance to a cluster member and all the non-cluster members
    The diffrence between the max distance to a cluster member and the min distance to a non-cluster member should be at least margin
    distance_fn: function that takes two tensors of shape (bs,tokens,latent_space_size) and returns a tensor of shape (bs,tokens,tokens)
    vectors: tensor of shape (bs,tokens,latent_space_size)
    token_mask: tensor of shape (bs,tokens)
    y: tensor of shape (bs,tokens)
    """
    orig_shape=y.shape
    distances, ic_distances, ec_distances, ic_mask,ec_mask = hac_compute_distances(distance_fn, vectors, token_mask,y) #((bs,tokens,tokens),(bs,tokens,tokens),(bs,tokens,tokens))
    #compute the diffrence between each ic_distance and each ec_distance per token
    sentence_losses=[]
    for s in range(orig_shape[0]):
        dist=distances[s,:,:]#(tokens,tokens)
        token_mask_s=token_mask[s,:]#(tokens,)
        cluster_mask=y[s,:]==1#(tokens,)
        ec_mask_s=(token_mask_s & ~cluster_mask).bool()#(tokens,)
        #if no ec tokens, no loss
        if ~torch.any(ec_mask_s):
            continue
        ic_distance_mask=dist[cluster_mask, :][:, cluster_mask] # (cluster_size, cluster_size)
        #if justa single cluster member, no loss
        if ic_distance_mask.shape[0]==1:
            continue
        upper_triangle_indices=torch.triu_indices(ic_distance_mask.shape[0], ic_distance_mask.shape[1], offset=1) # (2, cluster_size*(cluster_size-1)/2)
        ic_distance_mask_flattened=ic_distance_mask[upper_triangle_indices[0], upper_triangle_indices[1]] # (cluster_size*(cluster_size-1)/2)
        ec_distance_mask=dist[cluster_mask, :][:, ec_mask_s] # (cluster_size, num_tokens-cluster_size)
        ec_distance_mask_flattened=ec_distance_mask.flatten() # (cluster_size*(num_tokens-cluster_size))
        t1=ic_distance_mask_flattened
        t2=ec_distance_mask_flattened
        #extend the first tensor to the size of the second tensor
        t1_extended=t1.unsqueeze(1).expand(-1, t2.shape[0]) # (cluster_size*(cluster_size-1)/2, cluster_size*(num_tokens-cluster_size))
        #extend the second tensor to the size of the first tensor
        t2_extended=t2.unsqueeze(0).expand(t1.shape[0], -1) # (cluster_size*(cluster_size-1)/2, cluster_size*(num_tokens-cluster_size))
        penalty=t1_extended-t2_extended+margin # (cluster_size*(cluster_size-1)/2, cluster_size*(num_tokens-cluster_size))
        #flatten the penalty tensor
        penalty_flattened=penalty.flatten() # (cluster_size*(cluster_size-1)/2*cluster_size*(num_tokens-cluster_size))
        #mean of the positive values
        penalty_flattened=penalty_flattened[penalty_flattened>0]
        if penalty_flattened.shape[0]==0:
            continue
        loss=penalty_flattened[penalty_flattened>0].mean()
        sentence_losses.append(loss)
        #--------------------------------------
        # #penalize max(0,ic-ec+margin) for each ic and ec distances
        # ic_distance=ic_distances[s,:,:]#(tokens,tokens)
        # ec_distance=ec_distances[s,:,:]#(tokens,tokens)
        # token_mask_s=token_mask[s,:]#(tokens,)
        # cluster_mask=y[s,:]==1#(tokens,)
        # #get all ic_values that match the token mask and the cluster mask in a flattened tensor
        # ic_mask=token_mask_s & cluster_mask#(tokens,)
        # ic_distances_s=[]
        # ec_distances_s=[]
        # for ii in range(ic_mask.shape[0]):
        #     #for all other ic distances (starting in ii+1)
        #     if cluster_mask[ii]==1:
        #         for jj in range(ic_mask.shape[0]):
        #             #if is a cluser token
        #             if token_mask_s[jj]==1:
        #                 if cluster_mask[ii]==1 and cluster_mask[jj]==1 and ii<jj:
        #                     ic_distances_s.append(ic_distance[ii,jj])
        #                 else:
        #                     if cluster_mask[ii]==1 and cluster_mask[jj]==0:
        #                         ec_distances_s.append(ec_distance[ii,jj])
        # loss=[]
        # for ic in ic_distances_s:
        #     for ec in ec_distances_s:
        #         penalty=torch.max(torch.tensor([0.0],device=y.device),ic-ec+margin)
        #         if penalty>0:
        #             loss.append(penalty)
        # #average loss for the sentence
        # if len(loss)>0:
        #     loss=torch.mean(torch.stack(loss))
        #     sentence_losses.append(loss)
        #-------------------------------
        
        #limit the ic_values to the ones
        # for t in range(orig_shape[1]):
        #     if y[s,t]==0:
        #         continue
        #     ic_distance=ic_distances[s,t,:]#(tokens,)
        #     ec_distance=ec_distances[s,t,:]#(tokens,)
        #     v1=ec_distance
        #     v2=ic_distance
        #     vector1_expanded = v1.unsqueeze(1).expand(-1, len(v2)) # (tokens,tokens)
        #     vector2_expanded = v2.unsqueeze(0).expand(len(v1), -1) # (tokens,tokens)
        #     diff=vector1_expanded-vector2_expanded#(tokens,tokens)
        #     # if the token is not in a cluster, the loss should be at lest the margin, zero otherwise
        #     # if the token is in a cluster, the loss should be the difference

        #     for i in range(orig_shape[1]):
        #         if y[s,i]==0:
        #             if diff[t,i]<margin:
        #                 sum_loss.append(margin-diff[t,i])
        #         else:
        #             sum_loss.append(diff[t,i])
    if len(sentence_losses)==0:
        return torch.tensor(0.0,requires_grad=True,device=vectors.device),distances
    loss=torch.mean(torch.stack(sentence_losses))
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
    distances, ic_distances, ec_distances,_,_ = hac_compute_distances(distance_fn, vectors, token_mask,y)
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
    return distances,ic_distances,ec_distances,ic_mask,ec_mask

def cosine_distance(x,y):
    """
    cosine distance between two tensors
    """
    x_norm=normalize(x,dim=-1)
    y_norm=normalize(y,dim=-1)
    return 1-torch.matmul(x_norm,y_norm.transpose(-2,-1))