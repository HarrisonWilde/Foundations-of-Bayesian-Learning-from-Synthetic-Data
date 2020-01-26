# import arviz as az
import os
import sys
import numpy as np
import stan
from contextlib import contextmanager


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in Python, i.e. will suppress
    all print, even if the print originates in a compiled C/Fortran sub-function.
    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def merged_stderr_stdout():
    '''
    Use below stdout_redirected to also redirect stderr
    '''
    return stdout_redirected(to=sys.stdout, stdout=sys.stderr)


def run_experiment(file, warmup, samples, chains, init_cheat, y, indices, y_unseen, ytildes, scale, priormu, priora, priorb, beta, hp, w, beta_w, mu, sigma2, k):
    '''
    Uses Stan to perform MCMC sampling on the passed model, returns the resulting fit on passed data
    '''

    if init_cheat:
        init = [dict(mu=np.mean(y), sigma2=np.var(y))] * chains
    else:
        init = 'random'

    # Normal-Laplace with known scale and all data
    data = dict(n=len(y[~indices]), y1=y[~indices], m=len(y[indices]), y2=y[indices], j=len(y_unseen), y_unseen=y_unseen, k=len(ytildes),
                y_tildes=ytildes, mu_m=priormu, sig_p1=priora, sig_p2=priorb, hp=hp, scale=np.sqrt(2) * scale, beta=beta, beta_w=beta_w, w=w)
    model = stan.build(file, data)
    fit = model.sample(num_warmup=warmup, num_samples=samples, num_chains=chains, save_warmup=False)
    return fit
