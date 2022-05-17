from fcntl import DN_DELETE
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N==N_rep:
            target = target.repeat(N_rep//N,1)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

class TokenLabelSoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(TokenLabelSoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N==N_rep:
            target = target.repeat(N_rep//N,1)
        if len(target.shape)==3 and target.shape[-1]==2:
            ground_truth=target[:,:,0]
            target = target[:,:,1]
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

class TokenLabelCrossEntropy(nn.Module):
    """
    Token labeling loss.
    """
    def __init__(self, dense_weight=1.0, cls_weight = 1.0, mixup_active=True, classes = 1000, ground_truth = False):
        """
        Constructor Token labeling loss.
        """
        super(TokenLabelCrossEntropy, self).__init__()


        self.CE = SoftTargetCrossEntropy()

        self.dense_weight = dense_weight
        self.mixup_active = mixup_active
        self.classes = classes
        self.cls_weight = cls_weight
        self.ground_truth = ground_truth
        assert dense_weight+cls_weight>0


    def forward(self, x, target):
        output, aux_output, bb = x # dist_output is not used, output shape: [B, 1000]
        if len(bb) == 2:
            cx, cy = bb
        else:
            bbx1, bby1, bbx2, bby2 = bb

        B,N,C = aux_output.shape
        if len(target.shape)==2: # target : [B, 1000, 198] 
            target_cls=target
            target_aux = target.repeat(1,N).reshape(B*N,C)
        else: # default
            target_cls = target[:,:,1] # [:,:,0] : gt, [B, 1000, 1]: cls token, 
            if self.ground_truth: # default == False
                # use ground truth to help correct label.
                # rely more on ground truth if target_cls is incorrect.
                ground_truth = target[:,:,0] 
                ratio = (0.9 - 0.4 * (ground_truth.max(-1)[1] == target_cls.max(-1)[1])).unsqueeze(-1)
                target_cls = target_cls * ratio + ground_truth * (1 - ratio)
            target_aux = target[:,:,2:] # target label for tokens 
            target_aux = target_aux.transpose(1,2).reshape(-1,C)
        if len(bb) == 2:
            patch_size = math.sqrt(N)
            lam1 = cx * cy / N
            lam2 = (patch_size - cx) * cy / N
            lam3 = (patch_size - cy) * cx / N
            left_tensors = None
            try:
                target_cls_split = torch.stack([target_cls[i:i+4, :] for i in range(0, target_cls.shape[0], 4)], 0) # shape: [B/4, 4, 1000, 1]
            except:
                target_cls_split = torch.stack([target_cls[i:i+4, :] for i in range(0, target_cls.shape[0]-3, 4)], 0)
                left_tensors = target_cls_split[target_cls.shape[0]-3:]
            target_cls_stack = None
            for i in range(target_cls_split.shape[0]):
                target_tensor = target_cls_split[i]
                t1 = target_tensor
                t2 = target_tensor[[1,2,3,0], :]
                t3 = target_tensor[[2,3,0,1], :]
                t4 = target_tensor[[3,0,1,2], :]
                target_cls_tmp = lam1 * t1 + lam2 * t2 + lam3 * t3 + (1 - lam1 - lam2 - lam3) * t4 # [4, 1000]
                if i == 0:
                    target_cls_stack = target_cls_tmp
                else:
                    target_cls_stack = torch.cat([target_cls_stack, target_cls_tmp], 0)
            
            if left_tensors is not None:
                target_cls_stack = torch.cat([target_cls_stack, left_tensors], 0) # [B, 1000, 1]
            target_cls = target_cls_stack.cuda()
        else:
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / N)
            if lam<1:
                target_cls = lam*target_cls + (1-lam)*target_cls.flip(0)

        aux_output = aux_output.reshape(-1,C)
        loss_cls = self.CE(output, target_cls)
        loss_aux = self.CE(aux_output, target_aux)
        return self.cls_weight*loss_cls+self.dense_weight* loss_aux

