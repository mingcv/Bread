def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_conv_layers(model):
    cnt = 0
    for mo in model().modules():
        if type(mo).__name__ == 'Conv2d':
            cnt += 1

    print(model.__name__, cnt, count_parameters(model()))
