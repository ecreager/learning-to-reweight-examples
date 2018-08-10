import torch

def disparate_impact(y_logit, a, train=True):
    """disparate impact according to demographic parity; train specifies soft vs. hard classification"""
    if train:  # need differentiability; use soft DI 
        if y_logit.ndimension() == 1:
            yhat = torch.sigmoid(y_logit)
        else:
            yhat = torch.softmax(y_logit, 1)[:, 0]  # p(y = 0 | x)
    else:  # hard DI
        if y_logit.ndimension() == 1:
            yhat = y_logit > 0.
        else:
            _, yhat = torch.max(y_logit, 1)  # argmax_y p(y|x)
    #print(-1, y_logit)
    #print(0, train)
    #print(1, yhat)
    #print(2, a)
    #print(3, yhat[a==0])
    #print(33, yhat[a==0].float().mean())
    #print(4, yhat[a==1])
    #print(44, yhat[a==1].float().mean())
    return abs(torch.sub(
        yhat[a == 0].type(torch.float32).mean(),
        yhat[a == 1].type(torch.float32).mean()))


