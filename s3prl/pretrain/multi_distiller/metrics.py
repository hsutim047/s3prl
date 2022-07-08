import torch
import torch.nn as nn

def calculate_consine_similarity_score(preds):
    '''
    [B x N x T x D, B x N x T x D, ...]
    '''
    preds = [torch.flatten(pred, start_dim=2) for pred in preds]
    cos = nn.CosineSimilarity(dim=2)
    score = torch.zeros(preds[0].shape[0])
    for i in range(len(preds)):
        for j in range(len(preds)):
            if i == j:
                continue
            score += cos(preds[i], preds[j]).sum(dim=-1)
    return scores



if __name__ == "__main__":
    '''
    Eg: B=4, N=3, T=2, D=2
    '''
    preds = [torch.ones(4, 3, 2, 2), torch.ones(4, 3, 2, 2)]
    scores = calculate_consine_similarity_score(preds)
    print(scores)
