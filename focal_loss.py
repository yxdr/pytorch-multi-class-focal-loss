import torch


def focal_loss(outputs, targets, alpha_t=None, gamma=0.0):
    """

    :param outputs:
    :param targets:
    :param alpha_t: A list of weights for each class
    :param gamma:
    :return:
    """
    if alpha_t is None and gamma == 0:
        focal_loss = torch.nn.functional.cross_entropy(outputs, targets)

    elif alpha_t is not None and gamma == 0:
        focal_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                       weight=torch.tensor(alpha_t))

    elif alpha_t is None and gamma != 0:
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** gamma * ce_loss).mean()  # mean over the batch

    elif alpha_t is not None and gamma != 0:
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                    weight=torch.tensor(alpha_t), reduction='none')
        focal_loss = ((1 - p_t) ** gamma * ce_loss).mean()  # mean over the batch

    return focal_loss


if __name__ == '__main__':
    outputs = torch.tensor([[2, 1.],
                            [2.5, 1]])
    targets = torch.tensor([0, 1])

    print(torch.nn.functional.softmax(outputs, dim=1))

    print(focal_loss(outputs, targets, alpha_t=[0.5, 0.5], gamma=2.))
