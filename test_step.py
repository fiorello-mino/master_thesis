import torch
import torch.nn as nn
from src.parser import TrainingParser
from src.dataloaders import give_dataloaders, give_3D_dataloaders
from src.classes import ConvGRU, ConvGRU3D, ConvGRUClassifier
from src.utils import clip_series

parser = TrainingParser()
args = parser.parse_args()
args.paths = {"master": "."}

if not args.threeD and not args.extract_param:
    loaders = give_dataloaders(args)
    model_class = ConvGRU
elif args.extract_param:
    loaders = give_dataloaders(args)
    model_class = ConvGRUClassifier
else:
    loaders = give_3D_dataloaders(args)
    model_class = ConvGRU3D

batch = next(iter(loaders["train_set"]))
if args.num_params != 0:
    series, params = batch
else:
    series, params = batch, None

in_seq_length = min(args.subseq_min, series.shape[1] - 1)
future = series.shape[1] - in_seq_length - 1
x = clip_series(series, in_seq_length).to(args.device)

if not args.extract_param:
    out_ch = 1  # una sola fase scalare
else:
    out_ch = args.num_params

model = model_class(
    hidden_units=args.hidden,
    input_channels=1,
    output_channels=None if not args.extract_param else args.num_params,
    hidden_channels=args.channels,
    kernel_size=args.kernel_size,
    padding_mode=args.padding,
    bias=args.bias,
    divergence=args.divergence,
    conservative=args.conservative,
    num_params=args.num_params if not args.extract_param else 0,
    dropout=args.dropout,
    dropout_prob=args.dropout_prob,
).to(args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if args.extract_param:
    y = model(x, noisereg=0.0, approxinference=False)
    target = torch.cat([p.unsqueeze(1) for p in params], dim=1).float().to(args.device)
    loss_fn = nn.MSELoss()
else:
    y = model(x, future=future, params=params, noisereg=0.0, approxinference=False)
    target = series[:, 1:, ...].float().to(args.device)
    if not args.threeD:
        loss_fn = lambda a, b: nn.MSELoss()(a, b) + args.massW * nn.MSELoss()(
            torch.mean(a, dim=(-1, -2)), torch.mean(b, dim=(-1, -2))
        )
    else:
        loss_fn = lambda a, b: nn.MSELoss()(a, b) + args.massW * nn.MSELoss()(
            torch.mean(a, dim=(-1, -2, -3)), torch.mean(b, dim=(-1, -2, -3))
        )

loss = loss_fn(y, target)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)

print("step ok, loss =", loss.item())