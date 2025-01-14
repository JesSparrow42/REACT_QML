import torch


def fit(X, y, logger=None):
    with torch.no_grad():
        X = X.to(torch.float32)
        y = y.to(torch.float32)
        X = torch.cat((X, torch.ones_like(X[:, 0]).unsqueeze(1)), dim=1)

        A = torch.mm(X.T, X)
        A += torch.eye(A.shape[0], device=A.device) * 1e-2
        b = torch.mv(X.T, y)

        LinAlgError = torch._C._LinAlgError
        try:
            f = linalg_solve(A, b).flatten()
        except LinAlgError:
            if logger is not None:
                logger.print("matrix not invertible.")
            f = linalg_lstsq(A, b).flatten()

    return f[:-1]  # remove the bias


def linalg_solve(A, B):
    try:
        return torch.linalg.solve(A, B)
    except NotImplementedError:
        # Alternative method for Apple Silicon (MPS)
        return A.inverse() @ B


def linalg_lstsq(A, B):
    try:
        return torch.linalg.lstsq(A, B).solution
    except NotImplementedError:
        # Alternative method for Apple Silicon (MPS)
        return A.pinverse() @ B
