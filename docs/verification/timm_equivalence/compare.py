"""Compare the two fwdbwd.py dumps (timm 0.3.2 and timm 0.4.12): parameter names, initial weights, loss, gradients."""
import sys

import torch

a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
print("names equal", a["names"] == b["names"],
      "init maxdiff", (a["init"] - b["init"]).abs().max().item(),
      "loss equal", torch.equal(a["loss"], b["loss"]),
      "grad maxdiff", (a["grads"] - b["grads"]).abs().max().item(),
      "grads bitwise equal", torch.equal(a["grads"], b["grads"]))
