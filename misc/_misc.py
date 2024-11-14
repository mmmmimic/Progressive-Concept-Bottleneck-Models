import torch

class Logger:
    def __init__(self, log_dir, mode='a'):
        super().__init__()
        assert mode in ['w', 'a'], f"mode {mode} is neither 'w' nor 'a'"
        self.mode = mode
        self.log_dir = log_dir

    def fprint(self, log):
        log = str(log) if not isinstance(log, str) else log
        print(log)
        with open(self.log_dir, mode=self.mode) as f:        
            f.write(log)
            f.write("\n")


def to_gpu(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, dict):
        cuda_dict = {}
        for k, v in tensor.items():
            cuda_dict['k'] = v.to(device)
        del tensor
        return cuda_dict
    else:
        raise NotImplementedError()

def to_numpy(tensor):
    return tensor.squeeze().cpu().numpy()


def fit_line(x, y):
    N = float(len(x))
    if not N:
        return 0
    sx, sy, sxx, syy, sxy = 1e-9, 1e-9, 1e-9, 1e-9, 1e-9
    for i in range(0, int(N)):
        sx += x[i]
        sy += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    a = (sy*sx/N - sxy) / (sx*sx/N -sxx)
    return a