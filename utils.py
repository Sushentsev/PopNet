import torch

PAD = "[PAD]"
SOS = "[SOS]"
EOS = "[EOS]"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
