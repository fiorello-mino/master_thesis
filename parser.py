# <<< importing stuff <<<
import os
import time
from argparse import ArgumentParser
from warnings import warn
# --- importing stuff ---


class GeneralParser:
    """
    Parser generale: argomenti comuni a training, test, evaluation.
    """

    def __init__(self):
        self.parser = ArgumentParser()

        self.parser.add_argument(
            '--padding',
            type=str,
            default='circular',
            choices=['circular', 'zeros', 'reflect'],
            help='Padding mode for NN. For PBCs, use "circular".'
        )

        self.parser.add_argument(
            '--bias',
            action='store_true',
            help='Enable bias in convolutional layers.'
        )

        self.parser.add_argument(
            '--device',
            type=str,
            default='cuda',
            help='Device to use ("cpu" or "cuda[:n]").'
        )

        self.parser.add_argument(
            '--size',
            type=int,
            default=-1,
            help='Resize images to this size (pixels). -1 disables resizing.'
        )

        self.parser.add_argument(
            '--cmap',
            type=str,
            default='gray',
            help='Colormap for plotting and gifs.'
        )

        self.parser.add_argument(
            '--hidden',
            type=int,
            default=2,
            help='Number of hidden ConvGRU layers.'
        )

        self.parser.add_argument(
            '--channels',
            type=int,
            default=35,
            help='Number of channels in ConvGRU.'
        )

        self.parser.add_argument(
            '--kernel_size',
            type=int,
            default=3,
            help='Kernel size in ConvGRU convolutions.'
        )

        self.parser.add_argument(
            '--symm_kernel',
            action='store_true',
            help='Enforce bisymmetric kernels (rotational equivariance heuristic).'
        )

        self.parser.add_argument(
            '--seed',
            type=int,
            default=0,
            help='Random seed.'
        )

        self.parser.add_argument(
            '--nographics',
            action='store_true',
            help='Disable all graphical output (loss plots, png, gifs, vtk, npy).'
        )

        self.parser.add_argument(
            '--nogifs',
            action='store_true',
            help='Disable gif output (png still saved if graphics enabled).'
        )

        self.parser.add_argument(
            '--nocrop',
            action='store_true',
            help='Disable image cropping.'
        )

        self.parser.add_argument(
            '--croplims',
            type=float,
            nargs=2,
            default=(0.25, 0.75),
            help='Cropping bounds as fraction of image size (low, high).'
        )

        self.parser.add_argument(
            '--debug',
            action='store_true',
            help='Debug mode: only partial evaluation of datasets.'
        )

        self.parser.add_argument(
            '--id',
            type=str,
            default='',
            help='ID for this run (used in logs, model names, etc.).'
        )

        self.parser.add_argument(
            '--nproc',
            type=int,
            default=1,
            help='Number of worker processes (e.g. dataloader num_workers).'
        )

        self.parser.add_argument(
            '--divergence',
            action='store_true',
            help='Use divergence mode (continuity law dynamics).'
        )

        self.parser.add_argument(
            '--conservative',
            action='store_true',
            help='Conservative mode for non-divergence training (ignored if divergence=True).'
        )

        self.parser.add_argument(
            '--num_params',
            type=int,
            default=0,
            help='Number of external scalar parameters per sample.'
        )

        self.parser.add_argument(
            '--threeD',
            action='store_true',
            help='Enable 3D mode (requires .npy volumes). BETA.'
        )

        self.parser.add_argument(
            '--dropout',
            action='store_true',
            help='Enable dropout on ConvGRU hidden channels.'
        )

        self.parser.add_argument(
            '--dropout_prob',
            type=float,
            default=0.25,
            help='Dropout probability.'
        )

        self.parser.add_argument(
            '--extract_param',
            action='store_true',
            help='Training mode for parameter extraction (regression).'
        )

        self.parser.add_argument(
            '--rotation',
            action='store_true',
            help='Continuous rotation augmentation (overrides other rotations).'
        )

        self.parser.add_argument(
            '--rotation90',
            action='store_true',
            help='Discrete 90° rotations (overrides rotation_order).'
        )

        self.parser.add_argument(
            '--rotation_order',
            type=int,
            default=0,
            help='Custom rotational symmetry order (e.g. 3=triangular, 6=hexagonal).'
        )

        self.parser.add_argument(
            '--reflectionX',
            action='store_true',
            help='Reflection symmetry along x in data augmentation.'
        )

        self.parser.add_argument(
            '--reflectionY',
            action='store_true',
            help='Reflection symmetry along y in data augmentation.'
        )

        self.parser.add_argument(
            '--reflectionZ',
            action='store_true',
            help='Reflection symmetry along z (3D only).'
        )

        self.parser.add_argument(
            '--reflection',
            action='store_true',
            help='Reflection symmetry on all axes (overrides reflectionX/Y/Z).'
        )

        self.parser.add_argument(
            '--vtk',
            action='store_true',
            help='Save outputs in VTK format (if graphics enabled).'
        )

        self.parser.add_argument(
            '--npy',
            action='store_true',
            help='Save outputs in NPY format (if graphics enabled).'
        )

        self.parser.add_argument(
            '--compile',
            action='store_true',
            help='Enable torch.compile on the model (if available).'
        )

    def parse_args(self):
        args = self.parser.parse_args()

        # Graphics modes
        if args.nographics:
            args.graphics = False
            args.gifs = False
            args.vtk = False
            args.npy = False
        elif args.nogifs:
            args.graphics = True
            args.gifs = False
        else:
            args.graphics = True
            args.gifs = True

        # Reflection macro
        if args.reflection:
            args.reflectionX = True
            args.reflectionY = True
            args.reflectionZ = True

        # Crop flag
        args.crop = not args.nocrop

        # Device normalization
        if args.device.startswith('cuda'):
            try:
                import torch
                if not torch.cuda.is_available():
                    warn('CUDA requested but not available. Falling back to CPU.')
                    args.device = 'cpu'
            except Exception:
                args.device = 'cpu'

        # Dropout validation
        if args.dropout and not (0.0 <= args.dropout_prob < 1.0):
            raise ValueError('--dropout_prob must satisfy 0 <= p < 1 when --dropout is used.')

        return args


class TrainingParser(GeneralParser):
    """
    Parser per training: aggiunge epochs, lr, dataset paths etc.
    """

    def __init__(self):
        super().__init__()

        self.parser.add_argument(
            '--epochs',
            type=int,
            default=1_000,
            help='Number of training epochs.'
        )

        self.parser.add_argument(
            '--lr',
            type=float,
            default=5e-4,
            help='Base learning rate.'
        )

        self.parser.add_argument(
            '--batch',
            type=int,
            default=1,
            help='Batch size.'
        )

        self.parser.add_argument(
            '--weightd',
            type=float,
            default=0.0,
            help='Weight decay for Adam optimizer.'
        )

        self.parser.add_argument(
            '--massW',
            type=float,
            default=2.0,
            help='Weight of mass-conservation term in loss.'
        )

        self.parser.add_argument(
            '--translation',
            action='store_true',
            help='Enable translation augmentation.'
        )

        self.parser.add_argument(
            '--train_set',
            type=str,
            default='data/table_comb.txt',
            help='Path to training-set table file.'
        )

        self.parser.add_argument(
            '--valid_set',
            type=str,
            default='data/table_comb_valid.txt',
            help='Path to validation-set table file.'
        )

        self.parser.add_argument(
            '--subseq_min',
            type=int,
            default=1,
            help='Minimum subsequence length during training.'
        )

        self.parser.add_argument(
            '--subseq_max',
            type=int,
            default=99,
            help='Maximum subsequence length during training.'
        )

        self.parser.add_argument(
            '--logfreq',
            type=int,
            default=1,
            help='Logging frequency on terminal.'
        )

        self.parser.add_argument(
            '--dual',
            action='store_true',
            help='Use dual representation (phi and 1-phi).'
        )

        self.parser.add_argument(
            '--superbatch',
            type=int,
            default=1,
            help='Gradient accumulation factor (simulated batch size).'
        )

        self.parser.add_argument(
            '--bootstrap',
            action='store_true',
            help='Bootstrap resampling of train/valid sets.'
        )

        self.parser.add_argument(
            '--reload_model',
            type=str,
            default='',
            help='Path to .pt model to reload and continue training.'
        )

        self.parser.add_argument(
            '--twin_image',
            action='store_true',
            help='Twin-image mode: superimpose a second image (no overlap) to increase complexity.'
        )

        self.parser.add_argument(
            '--ramp',
            action='store_true',
            help='Enable linear ramp for teacher-forcing length.'
        )

        self.parser.add_argument(
            '--ramp_length',
            type=int,
            default=100,
            help='Epochs to reach full BPTT in ramp mode.'
        )

        self.parser.add_argument(
            '--start_ramp',
            type=int,
            default=0,
            help='Ramp offset (effective ramp length = ramp_length - start_ramp).'
        )

        self.parser.add_argument(
            '--noise_reg',
            type=float,
            default=0.0,
            help='Stddev of Gaussian noise added during training (regularization).'
        )

    def parse_args(self):
        args = super().parse_args()

        # Scale lr by superbatch (gradient accumulation)
        args.lr /= args.superbatch

        # Reload flag
        args.reload = (args.reload_model != '')

        return args


class NonTrainParser(GeneralParser):
    """
    Parser per task non-di-training (evaluation, test, ecc.).
    """

    def __init__(self):
        super().__init__()

        self.parser.add_argument(
            '--tot_frames',
            type=int,
            default=100,
            help='Total number of frames in evolution.'
        )

        self.parser.add_argument(
            '--in_frames',
            type=int,
            default=1,
            help='Number of initial frames provided.'
        )

        self.parser.add_argument(
            '--model_name',
            type=str,
            default='models/model.pt',
            help='Path to model. If a folder, all .pt will form a committee.'
        )

        self.parser.add_argument(
            '--out_all',
            action='store_true',
            help='Output all frames as png.'
        )

        self.parser.add_argument(
            '--AR',
            action='store_true',
            help='Estimate aspect ratio of evolving configuration (simple shapes).'
        )


class EvaluationParser(NonTrainParser):
    """
    Parser per valutazione / generazione.
    """

    def __init__(self):
        super().__init__()

        self.parser.add_argument(
            '--init_geo',
            type=str,
            default='',
            help='Path to .py defining initial geometry (shapelist_add / shapelist_remove).'
        )

        self.parser.add_argument(
            '--load_image',
            type=str,
            default='',
            help='Path to initial condition image.'
        )

        self.parser.add_argument(
            '--scatter',
            action='store_true',
            help='Scatter mode: save outputs of committee members separately.'
        )

        self.parser.add_argument(
            '--dual',
            action='store_true',
            help='Use dual dynamics for imported image / geometry.'
        )

        self.parser.add_argument(
            '--params_list',
            type=float,
            nargs='+',
            help='List of parameters passed to predictor (consistent with num_params).'
        )

        self.parser.add_argument(
            '--save_every',
            type=int,
            default=1,
            help='Saving frequency for generated states.'
        )

    def parse_args(self):
        args = super().parse_args()

        if args.init_geo and args.load_image:
            args.gengeo = False
            warn(
                'Both geometry init file and image init file were specified. '
                'Falling back on image init. Remove --load_image to use init_geo.'
            )
        elif args.init_geo:
            args.gengeo = True
        elif args.load_image:
            args.gengeo = False
        else:
            raise RuntimeError('An initial condition was not specified.')

        return args


class TestParser(NonTrainParser):
    """
    Parser per testing di modelli.
    """

    def __init__(self):
        super().__init__()

        self.parser.add_argument(
            '--test_set',
            type=str,
            default='data/table_comb_test.txt',
            help='Path to test-set table file.'
        )

        self.parser.add_argument(
            '--translation',
            action='store_true',
            help='Enable translation augmentation in test.'
        )

        self.parser.add_argument(
            '--dual_prob',
            type=float,
            default=0.5,
            help='Probability of using dual dynamics in test.'
        )

        self.parser.add_argument(
            '--batch',
            type=int,
            default=1,
            help='Batch size (only >1 when extract_param is True).'
        )

    def parse_args(self):
        args = super().parse_args()

        if not args.extract_param and args.batch > 1:
            print('Resetting batch to 1 for evolution testing (extract_param=False).')
            time.sleep(1)
            args.batch = 1

        return args# <<< importing stuff <<<
import os
import time
from argparse import ArgumentParser
from warnings import warn
# --- importing stuff ---


class GeneralParser:
    """
    Parser generale: argomenti comuni a training, test, evaluation.
    """

    def __init__(self):
        self.parser = ArgumentParser()

        self.parser.add_argument(
            '--padding',
            type=str,
            default='circular',
            choices=['circular', 'zeros', 'reflect'],
            help='Padding mode for NN. For PBCs, use "circular".'
        )

        self.parser.add_argument(
            '--bias',
            action='store_true',
            help='Enable bias in convolutional layers.'
        )

        self.parser.add_argument(
            '--device',
            type=str,
            default='cuda',
            help='Device to use ("cpu" or "cuda[:n]").'
        )

        self.parser.add_argument(
            '--size',
            type=int,
            default=-1,
            help='Resize images to this size (pixels). -1 disables resizing.'
        )

        self.parser.add_argument(
            '--cmap',
            type=str,
            default='gray',
            help='Colormap for plotting and gifs.'
        )

        self.parser.add_argument(
            '--hidden',
            type=int,
            default=2,
            help='Number of hidden ConvGRU layers.'
        )

        self.parser.add_argument(
            '--channels',
            type=int,
            default=35,
            help='Number of channels in ConvGRU.'
        )

        self.parser.add_argument(
            '--kernel_size',
            type=int,
            default=3,
            help='Kernel size in ConvGRU convolutions.'
        )

        self.parser.add_argument(
            '--symm_kernel',
            action='store_true',
            help='Enforce bisymmetric kernels (rotational equivariance heuristic).'
        )

        self.parser.add_argument(
            '--seed',
            type=int,
            default=0,
            help='Random seed.'
        )

        self.parser.add_argument(
            '--nographics',
            action='store_true',
            help='Disable all graphical output (loss plots, png, gifs, vtk, npy).'
        )

        self.parser.add_argument(
            '--nogifs',
            action='store_true',
            help='Disable gif output (png still saved if graphics enabled).'
        )

        self.parser.add_argument(
            '--nocrop',
            action='store_true',
            help='Disable image cropping.'
        )

        self.parser.add_argument(
            '--croplims',
            type=float,
            nargs=2,
            default=(0.25, 0.75),
            help='Cropping bounds as fraction of image size (low, high).'
        )

        self.parser.add_argument(
            '--debug',
            action='store_true',
            help='Debug mode: only partial evaluation of datasets.'
        )

        self.parser.add_argument(
            '--id',
            type=str,
            default='',
            help='ID for this run (used in logs, model names, etc.).'
        )

        self.parser.add_argument(
            '--nproc',
            type=int,
            default=1,
            help='Number of worker processes (e.g. dataloader num_workers).'
        )

        self.parser.add_argument(
            '--divergence',
            action='store_true',
            help='Use divergence mode (continuity law dynamics).'
        )

        self.parser.add_argument(
            '--conservative',
            action='store_true',
            help='Conservative mode for non-divergence training (ignored if divergence=True).'
        )

        self.parser.add_argument(
            '--num_params',
            type=int,
            default=0,
            help='Number of external scalar parameters per sample.'
        )

        self.parser.add_argument(
            '--threeD',
            action='store_true',
            help='Enable 3D mode (requires .npy volumes). BETA.'
        )

        self.parser.add_argument(
            '--dropout',
            action='store_true',
            help='Enable dropout on ConvGRU hidden channels.'
        )

        self.parser.add_argument(
            '--dropout_prob',
            type=float,
            default=0.25,
            help='Dropout probability.'
        )

        self.parser.add_argument(
            '--extract_param',
            action='store_true',
            help='Training mode for parameter extraction (regression).'
        )

        self.parser.add_argument(
            '--rotation',
            action='store_true',
            help='Continuous rotation augmentation (overrides other rotations).'
        )

        self.parser.add_argument(
            '--rotation90',
            action='store_true',
            help='Discrete 90° rotations (overrides rotation_order).'
        )

        self.parser.add_argument(
            '--rotation_order',
            type=int,
            default=0,
            help='Custom rotational symmetry order (e.g. 3=triangular, 6=hexagonal).'
        )

        self.parser.add_argument(
            '--reflectionX',
            action='store_true',
            help='Reflection symmetry along x in data augmentation.'
        )

        self.parser.add_argument(
            '--reflectionY',
            action='store_true',
            help='Reflection symmetry along y in data augmentation.'
        )

        self.parser.add_argument(
            '--reflectionZ',
            action='store_true',
            help='Reflection symmetry along z (3D only).'
        )

        self.parser.add_argument(
            '--reflection',
            action='store_true',
            help='Reflection symmetry on all axes (overrides reflectionX/Y/Z).'
        )

        self.parser.add_argument(
            '--vtk',
            action='store_true',
            help='Save outputs in VTK format (if graphics enabled).'
        )

        self.parser.add_argument(
            '--npy',
            action='store_true',
            help='Save outputs in NPY format (if graphics enabled).'
        )

        self.parser.add_argument(
            '--compile',
            action='store_true',
            help='Enable torch.compile on the model (if available).'
        )

    def parse_args(self):
        args = self.parser.parse_args()

        # Graphics modes
        if args.nographics:
            args.graphics = False
            args.gifs = False
            args.vtk = False
            args.npy = False
        elif args.nogifs:
            args.graphics = True
            args.gifs = False
        else:
            args.graphics = True
            args.gifs = True

        # Reflection macro
        if args.reflection:
            args.reflectionX = True
            args.reflectionY = True
            args.reflectionZ = True

        # Crop flag
        args.crop = not args.nocrop

        # Device normalization
        if args.device.startswith('cuda'):
            try:
                import torch
                if not torch.cuda.is_available():
                    warn('CUDA requested but not available. Falling back to CPU.')
                    args.device = 'cpu'
            except Exception:
                args.device = 'cpu'

        # Dropout validation
        if args.dropout and not (0.0 <= args.dropout_prob < 1.0):
            raise ValueError('--dropout_prob must satisfy 0 <= p < 1 when --dropout is used.')

        return args


class TrainingParser(GeneralParser):
    """
    Parser per training: aggiunge epochs, lr, dataset paths etc.
    """

    def __init__(self):
        super().__init__()

        self.parser.add_argument(
            '--epochs',
            type=int,
            default=1_000,
            help='Number of training epochs.'
        )

        self.parser.add_argument(
            '--lr',
            type=float,
            default=5e-4,
            help='Base learning rate.'
        )

        self.parser.add_argument(
            '--batch',
            type=int,
            default=1,
            help='Batch size.'
        )

        self.parser.add_argument(
            '--weightd',
            type=float,
            default=0.0,
            help='Weight decay for Adam optimizer.'
        )

        self.parser.add_argument(
            '--massW',
            type=float,
            default=2.0,
            help='Weight of mass-conservation term in loss.'
        )

        self.parser.add_argument(
            '--translation',
            action='store_true',
            help='Enable translation augmentation.'
        )

        self.parser.add_argument(
            '--train_set',
            type=str,
            default='data/table_comb.txt',
            help='Path to training-set table file.'
        )

        self.parser.add_argument(
            '--valid_set',
            type=str,
            default='data/table_comb_valid.txt',
            help='Path to validation-set table file.'
        )

        self.parser.add_argument(
            '--subseq_min',
            type=int,
            default=1,
            help='Minimum subsequence length during training.'
        )

        self.parser.add_argument(
            '--subseq_max',
            type=int,
            default=99,
            help='Maximum subsequence length during training.'
        )

        self.parser.add_argument(
            '--logfreq',
            type=int,
            default=1,
            help='Logging frequency on terminal.'
        )

        self.parser.add_argument(
            '--dual',
            action='store_true',
            help='Use dual representation (phi and 1-phi).'
        )

        self.parser.add_argument(
            '--superbatch',
            type=int,
            default=1,
            help='Gradient accumulation factor (simulated batch size).'
        )

        self.parser.add_argument(
            '--bootstrap',
            action='store_true',
            help='Bootstrap resampling of train/valid sets.'
        )

        self.parser.add_argument(
            '--reload_model',
            type=str,
            default='',
            help='Path to .pt model to reload and continue training.'
        )

        self.parser.add_argument(
            '--twin_image',
            action='store_true',
            help='Twin-image mode: superimpose a second image (no overlap) to increase complexity.'
        )

        self.parser.add_argument(
            '--ramp',
            action='store_true',
            help='Enable linear ramp for teacher-forcing length.'
        )

        self.parser.add_argument(
            '--ramp_length',
            type=int,
            default=100,
            help='Epochs to reach full BPTT in ramp mode.'
        )

        self.parser.add_argument(
            '--start_ramp',
            type=int,
            default=0,
            help='Ramp offset (effective ramp length = ramp_length - start_ramp).'
        )

        self.parser.add_argument(
            '--noise_reg',
            type=float,
            default=0.0,
            help='Stddev of Gaussian noise added during training (regularization).'
        )

    def parse_args(self):
        args = super().parse_args()

        # Scale lr by superbatch (gradient accumulation)
        args.lr /= args.superbatch

        # Reload flag
        args.reload = (args.reload_model != '')

        return args


class NonTrainParser(GeneralParser):
    """
    Parser per task non-di-training (evaluation, test, ecc.).
    """

    def __init__(self):
        super().__init__()

        self.parser.add_argument(
            '--tot_frames',
            type=int,
            default=100,
            help='Total number of frames in evolution.'
        )

        self.parser.add_argument(
            '--in_frames',
            type=int,
            default=1,
            help='Number of initial frames provided.'
        )

        self.parser.add_argument(
            '--model_name',
            type=str,
            default='models/model.pt',
            help='Path to model. If a folder, all .pt will form a committee.'
        )

        self.parser.add_argument(
            '--out_all',
            action='store_true',
            help='Output all frames as png.'
        )

        self.parser.add_argument(
            '--AR',
            action='store_true',
            help='Estimate aspect ratio of evolving configuration (simple shapes).'
        )


class EvaluationParser(NonTrainParser):
    """
    Parser per valutazione / generazione.
    """

    def __init__(self):
        super().__init__()

        self.parser.add_argument(
            '--init_geo',
            type=str,
            default='',
            help='Path to .py defining initial geometry (shapelist_add / shapelist_remove).'
        )

        self.parser.add_argument(
            '--load_image',
            type=str,
            default='',
            help='Path to initial condition image.'
        )

        self.parser.add_argument(
            '--scatter',
            action='store_true',
            help='Scatter mode: save outputs of committee members separately.'
        )

        self.parser.add_argument(
            '--dual',
            action='store_true',
            help='Use dual dynamics for imported image / geometry.'
        )

        self.parser.add_argument(
            '--params_list',
            type=float,
            nargs='+',
            help='List of parameters passed to predictor (consistent with num_params).'
        )

        self.parser.add_argument(
            '--save_every',
            type=int,
            default=1,
            help='Saving frequency for generated states.'
        )

    def parse_args(self):
        args = super().parse_args()

        if args.init_geo and args.load_image:
            args.gengeo = False
            warn(
                'Both geometry init file and image init file were specified. '
                'Falling back on image init. Remove --load_image to use init_geo.'
            )
        elif args.init_geo:
            args.gengeo = True
        elif args.load_image:
            args.gengeo = False
        else:
            raise RuntimeError('An initial condition was not specified.')

        return args


class TestParser(NonTrainParser):
    """
    Parser per testing di modelli.
    """

    def __init__(self):
        super().__init__()

        self.parser.add_argument(
            '--test_set',
            type=str,
            default='data/table_comb_test.txt',
            help='Path to test-set table file.'
        )

        self.parser.add_argument(
            '--translation',
            action='store_true',
            help='Enable translation augmentation in test.'
        )

        self.parser.add_argument(
            '--dual_prob',
            type=float,
            default=0.5,
            help='Probability of using dual dynamics in test.'
        )

        self.parser.add_argument(
            '--batch',
            type=int,
            default=1,
            help='Batch size (only >1 when extract_param is True).'
        )

    def parse_args(self):
        args = super().parse_args()

        if not args.extract_param and args.batch > 1:
            print('Resetting batch to 1 for evolution testing (extract_param=False).')
            time.sleep(1)
            args.batch = 1

        return args