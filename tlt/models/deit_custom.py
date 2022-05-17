import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.resnet import resnet26d, resnet50d, resnet101d
from timm.models.vision_transformer import VisionTransformer, _cfg


from .layers import *

__all__ = [ "deit_small_custom"]

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        'mix_token' : True,
        'mosaic_token' : False,
        **kwargs
    }

default_cfgs = {
    'deit_small_custom': _cfg(),
    'deit_small_custom_mosaic': _cfg(mix_token=False, mosaic_token=True)
}

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def bbox_mosaic(W, H, mosaic_index, cx, cy):
    if mosaic_index == 0:
        x1, y1, x2, y2 = 0, 0, cx, cy # left top
    elif mosaic_index == 1:
        x1, y1, x2, y2 = cx, 0, W, cy # right top
    elif mosaic_index == 2:
        x1, y1, x2, y2 = 0, cy, cx, H # left bottom
    else:
        x1, y1, x2, y2 = cx, cy, W, H # right bottom

    return (x1, y1, x2, y2)


def get_dpr(drop_path_rate,depth,drop_path_decay='linear'):
    if drop_path_decay=='linear':
        # linear dpr decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    elif drop_path_decay=='fix':
        # use fixed dpr
        dpr= [drop_path_rate]*depth
    else:
        # use predefined drop_path_rate list
        assert len(drop_path_rate)==depth
        dpr=drop_path_rate
    return dpr

def get_block(block_type, **kargs):
    if block_type=='mha':
        # multi-head attention block
        return MHABlock(**kargs)
    elif block_type=='ffn':
        # feed forward block
        return FFNBlock(**kargs)
    elif block_type=='tr':
        # transformer block
        return Block(**kargs)

class DeiT_custom(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., drop_path_decay='linear', hybrid_backbone=None, norm_layer=nn.LayerNorm, p_emb='4_2', head_dim = None,
                 skip_lam = 1.0,order=None, mix_token=False, mosaic_token=False, return_dense=False):
        super().__init__()
        # for lvvit
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.output_dim = embed_dim if num_classes==0 else num_classes
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            if p_emb=='4_2':
                patch_embed_fn = PatchEmbed4_2
            elif p_emb=='4_2_128':
                patch_embed_fn = PatchEmbed4_2_128
            else:
                patch_embed_fn = PatchEmbedNaive

            self.patch_embed = patch_embed_fn(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        
        # for deit
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        
        
        if order is None:
            dpr=get_dpr(drop_path_rate, depth, drop_path_decay)
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam)
                for i in range(depth)])
        else:
            # use given order to sequentially generate modules
            dpr=get_dpr(drop_path_rate, len(order), drop_path_decay)
            self.blocks = nn.ModuleList([
                get_block(order[i],
                    dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam)
                for i in range(len(order))])
        
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.mix_token=mix_token
        self.mosaic_token = mosaic_token
        self.return_dense=return_dense
    
        if return_dense:
            self.aux_head=nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            
        if mix_token:
            self.beta = 1.0
            assert return_dense, "always return all features when mixtoken is enabled"
            
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, GroupLinear):
            trunc_normal_(m.group_weight, std=.02)
            if isinstance(m, GroupLinear) and m.group_bias is not None:
                nn.init.constant_(m.group_bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
    def forward_embeddings(self,x):
        x = self.patch_embed(x)
        return x
    
    def forward_tokens(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_features(self,x):
        # simple forward to obtain feature map (without mixtoken)
        x = self.forward_embeddings(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.forward_tokens(x)
        return x

    def mixtoken(self, x):
        lam = np.random.beta(self.beta, self.beta)
        patch_h, patch_w = x.shape[2],x.shape[3]
        self.bbx1, self.bby1, self.bbx2, self.bby2 = rand_bbox(x.size(), lam)
        temp_x = x.clone()
        temp_x[:, :, self.bbx1:self.bbx2, self.bby1:self.bby2] = x.flip(0)[:, :, self.bbx1:self.bbx2, self.bby1:self.bby2] # flip(0) : batch 기준으로 뒤집음.(4번배치가 0번으로, 3번배치가 1번으로...)
        return temp_x, patch_h, patch_w
    
    def mosaictoken(self, x):
        left_tensors = None
        patch_h, patch_w = x.shape[2],x.shape[3]
        try:
            sliced_tensors = torch.stack([x[i:i+4, :, :, :] for i in range(0, x.shape[0], 4)], 0) # shape: [B/4, 4, 384, 14, 14]
        except: # if len(tensor) % 4 != 0 
            sliced_tensors = torch.stack([x[i:i+4, :, :, :] for i in range(0, x.shape[0]-3, 4)], 0)
            left_tensors = x[x.shape[0]-3:]
        mosaic_output = None
        
        # uniform
        W = x.shape[2]
        H = x.shape[3]
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        for i in range(sliced_tensors.shape[0]):
            mosaic_tensor = sliced_tensors[i]
            m1 = mosaic_tensor
            m2 = mosaic_tensor[[1,2,3,0], :, :, :]
            m3 = mosaic_tensor[[2,3,0,1], :, :, :]
            m4 = mosaic_tensor[[3,0,1,2], :, :, :]
            m_list = [m1, m2, m3, m4]
            aggregate = torch.zeros(mosaic_tensor.shape)
            
            for m_idx, m_tensor in enumerate(m_list):
                self.bbx1, self.bby1, self.bbx2, self.bby2 = bbox_mosaic(patch_w, patch_h, m_idx, cx, cy)
                aggregate[:, :, self.bbx1:self.bbx2, self.bby1:self.bby2] = m_tensor[:,:,self.bbx1:self.bbx2, self.bby1:self.bby2]
            if i == 0:
                mosaic_output = aggregate
            else:
                mosaic_output = torch.cat([mosaic_output, aggregate])
        if left_tensors is not None:
            mosaic_output = torch.cat([mosaic_output, left_tensors])   
        mosaic_output = mosaic_output.cuda() 
        return mosaic_output, patch_h, patch_w, (cx, cy)
    
    def forward(self, x):
        x = self.forward_embeddings(x) # output: [batch, 384(embed_dim), 14(patch size), 14]
        # token level mixtoken augmentation 
        if self.mix_token and self.training:
            mixed_x, patch_h, patch_w = self.mixtoken(x)
            x = mixed_x
        elif self.mosaic_token and self.training:
            mosaic_x, patch_h, patch_w, (cx, cy) = self.mosaictoken(x)
            x = mosaic_x
        else:
            self.bbx1, self.bby1, self.bbx2, self.bby2 = 0,0,0,0

        x = x.flatten(2).transpose(1, 2) # [batch, 14*14, 384]
        x = self.forward_tokens(x)
        x_cls = self.head(x[:, 0])
        x_dist = self.head_dist(x[:, 1])
        
        if self.return_dense:
            x_aux = self.aux_head(x[:,2:]) # [B, 196, 1000]
            if not self.training:
                return (x_cls + x_dist) / 2 + 0.5*x_aux.max(1)[0]
            
            # recover the mixed part
            if self.mix_token and self.training:
                x_aux = x_aux.reshape(x_aux.shape[0],patch_h, patch_w,x_aux.shape[-1]) # [B, 14, 14, 1000]
                mixed_x = x_aux.clone()
                mixed_x[:, self.bbx1:self.bbx2, self.bby1:self.bby2, :] = x_aux.flip(0)[:, self.bbx1:self.bbx2, self.bby1:self.bby2, :] 
                x_aux = mixed_x
                x_aux = x_aux.reshape(x_aux.shape[0],patch_h*patch_w,x_aux.shape[-1]) # [B, 196, 1000]
                
            # recover the mixed part
            if self.mosaic_token and self.training:
                x_aux = x_aux.reshape(x_aux.shape[0],patch_h, patch_w,x_aux.shape[-1]) # [B, 14, 14, 1000]
                mosaic_x = None
                left_tensors = None
                try:
                    sliced_tensors = torch.stack([x_aux[i:i+4, :, :, :] for i in range(0, x_aux.shape[0], 4)], 0) # shape: [B/4, 4, 384, 14, 14]
                except: # if len(tensor) % 4 != 0 
                    sliced_tensors = torch.stack([x_aux[i:i+4, :, :, :] for i in range(0, x_aux.shape[0]-3, 4)], 0)
                    left_tensors = x_aux[x_aux.shape[0]-3:]
                
                for i in range(sliced_tensors.shape[0]):
                    mosaic_tensor = sliced_tensors[i]
                    m1 = mosaic_tensor
                    m2 = mosaic_tensor[[1,2,3,0], :, :, :]
                    m3 = mosaic_tensor[[2,3,0,1], :, :, :]
                    m4 = mosaic_tensor[[3,0,1,2], :, :, :]
                    m_list = [m1, m4, m3, m2] # reversed order
                    aggregate = torch.zeros(mosaic_tensor.shape)
            
                    for m_idx, m_tensor in enumerate(m_list):
                        self.bbx1, self.bby1, self.bbx2, self.bby2 = bbox_mosaic(patch_w, patch_h, m_idx, cx, cy)
                        aggregate[:, :, self.bbx1:self.bbx2, self.bby1:self.bby2] = m_tensor[:,:,self.bbx1:self.bbx2, self.bby1:self.bby2]
                    if i == 0:
                        mosaic_x = aggregate
                    else:
                        mosaic_x = torch.cat([mosaic_x, aggregate], 0)
                if left_tensors is not None:
                    mosaic_x = torch.cat([mosaic_x, left_tensors], 0) 
                x_aux = mosaic_x.cuda()
                x_aux = x_aux.reshape(x_aux.shape[0],patch_h*patch_w,x_aux.shape[-1]) # [B, 196, 1000]
                
                return x_cls, x_aux, (cx, cy)
            
            return x_cls, x_aux, (self.bbx1, self.bby1, self.bbx2, self.bby2)
            
        else:
            return (x_cls + x_dist) / 2
        


        # if self.return_dense:
        #     x_aux = self.aux_head(x[:,1:])
        #     if not self.training:
        #         return x_cls+0.5*x_aux.max(1)[0]

        #     # recover the mixed part
        #     if self.mix_token and self.training:
        #         x_aux = x_aux.reshape(x_aux.shape[0],patch_h, patch_w,x_aux.shape[-1])
        #         temp_x = x_aux.clone()
        #         temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
        #         x_aux = temp_x
        #         x_aux = x_aux.reshape(x_aux.shape[0],patch_h*patch_w,x_aux.shape[-1])

        #     return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)
        # return x_cls
        

@register_model
def deit_small_custom(pretrained=False, **kwargs):
    model = DeiT_custom(
        patch_size=16, embed_dim=384, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), mix_token=True, mosaic_token=False, return_dense=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_small_custom_mosaic(pretrained=False, **kwargs):
    model = DeiT_custom(
        patch_size=16, embed_dim=384, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), mix_token=False, mosaic_token=True, return_dense=True, **kwargs)
    model.default_cfg = default_cfgs['deit_small_custom_mosaic']
    return model