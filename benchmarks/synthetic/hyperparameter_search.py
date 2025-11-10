import os
import subprocess

lrs = [1e-2, 1e-3, 1e-4]
gammas = [1e-2, 1e-3, 1e-4]
betas = [1e-1, 1e-2, 1e-3]

for lr in lrs:
    for gamma in gammas:
        for beta in betas:
            command = [
                'python3',
                'benchmarks/synthetic/main.py',
                '--optimizer', 'belavkin',
                '--lr', str(lr),
                '--gamma', str(gamma),
                '--beta', str(beta),
                '--epochs', '200'
            ]
            subprocess.run(command)
