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

model = model_class(
    hiddenunits=args.hidden,
    inputchannels=1,
    outputchannels=None if not args.extract_param else args.num_params,
    hiddenchannels=args.channels,
    kernelsize=args.kernel_size,
    paddingmode=args.padding,
    bias=args.bias,
    divergence=args.divergence,
    conservative=args.conservative,
    numparams=args.num_params if not args.extract_param else 0,
    dropout=args.dropout,
    dropoutprob=args.dropout_prob,
).to(args.device)

if args.extract_param:
    y = model(x, noisereg=0.0, approxinference=False)
else:
    y = model(x, future=future, params=params, noisereg=0.0, approxinference=False)

print("input:", x.shape)
print("output:", y.shape)