from src.parser import TrainingParser
from src.dataloaders import give_dataloaders, give_3D_dataloaders

parser = TrainingParser()
args = parser.parse_args()
args.paths = {"master": "."}

if not args.threeD:
    loaders = give_dataloaders(args)
else:
    loaders = give_3D_dataloaders(args)

batch = next(iter(loaders["train_set"]))

if args.num_params != 0:
    series, params = batch
    print("series shape:", series.shape)
    print("num params:", len(params))
    print("first param batch len:", len(params[0]))
else:
    series = batch
    print("series shape:", series.shape)