import torch
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

if args.extract_param:
    y = model(x, noisereg=0.0, approxinference=False)
else:
    y = model(x, future=future, params=params, noisereg=0.0, approxinference=False)

print("input:", x.shape)
print("output:", y.shape)