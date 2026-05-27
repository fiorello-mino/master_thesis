#!/usr/bin/env python3
import argparse
from pathlib import Path


HEADER = """info:                                                   100

dimension of world:                                          2
smallest number:                                             1.e-6

name:\tsurf
#include "init/config.ini"

degree:                                                      10

axis->x:                                                     [ 1, 1, 0]
axis->y:                                                     [ -1, 1, 0]
axis->z:                                                     [ 0, 0, 1]

surf->space->components:                                2
surf->space->components->phi:\t\t\t\t1\t% 1->Yes
surf->space->components->mu:\t\t\t1\t% 1->Yes
surf->space->components->kappa:\t\t\t0\t% 1->Yes
surf->space->name:\t\t\t\t\t[phi,mu]

#############################################################################################
#\tCUSTOM
surf->stabilizing function for mu:\t\t\t1\t% 1->Yes 0->No
surf->eps:                                              0.078125

#       BC
surf->boundary conditions->Periodic BC:\t\t\t  [ -1 ]
surf->boundary conditions->contact angle->boundary index: [ ]
surf->boundary conditions->contact angle->theta:          [ ] % degree

#       PHYSICS
surf->physics->energy->gamma:\t\t\t\t1
surf->physics->energy->gamma->anisotropy->mode:             no
    surf->physics->energy->gamma->anisotropy->output:        ani_gamma.dat
    surf->physics->energy->gamma->anisotropy->baseline:      1.
% ms -> gamma=MS function(n)
    surf->physics->energy->gamma->anisotropy->ms->file:      init/GammaGe_abs.dat
    surf->physics->energy->gamma->anisotropy->ms->absolute values:             1

surf->physics->energy->corner:\t\t\t\t0.

surf->physics->mobility:\t\t\t\t5e-5


surf->physics->ac->k:\t\t\t\t\t0
surf->growth->rate:                                     0.
surf->growth->flux distribution:                        0       % 0-Isotropic 1-Directional
surf->growth->flux direction:                           [ 0, 0, -1 ]
surf->growth->anisotropy:                               no %init/cube.dat   % Specify file name or 0->No

############################################################################################
#############################################################################################
#       PHI

surf->phi->mode:                                shape % external file , constant
surf->phi->external file:\t\t\t\tinit/test.arh
surf->phi->constant:                            1.
"""

FOOTER = """
#############################################################################################
surf->phi->signed distance:\t\t\t                                1          % 1->Yes 0->No
surf->phi->signed distance->tolerance: \t\t\t\t        1.e-4
surf->phi->signed distance->maximal number of iteration steps: \t        100
surf->phi->signed distance->Gauss-Seidel iteration: \t\t\t1
surf->phi->signed distance->infinity value: \t\t\t\t1.e8
surf->phi->signed distance->boundary initialization: \t\t\t3

#\tLOCAL REFINEMENT
surf->phi->refinement:                      \t\tpf     % grad, pf
surf->phi->refinement->pf->level in outer domain: \t4
surf->phi->refinement->pf->level on interface: \t\t12
surf->phi->refinement->pf->level on points: \t\t12
surf->phi->refinement->pf->level in inner domain: \t4
surf->phi->refinement->pf->initial level:               4
surf->phi->refinement->pf->min interface value:\t\t0.05    % Default=0.05
surf->phi->refinement->pf->max interface value:\t\t0.95    % Default=0.95

#############################################################################################
#\tOUTPUT
output->directory:                                      pore_2D
surf->output->filename:                                 surf
surf->output->check:                                    check
surf->output->physics:                                  phys
surf->output->last:\t                                last
surf->output->last->iterations:                  100

surf->output->write every i-th timestep:         1
surf->output->ParaView format:                          1
surf->output->ParaView animation:                       1
surf->output->append index:                             1
surf->output->index length:                             7
surf->output->index decimals:                           6
surf->output->ARH format:                               0
surf->output->ARH2 format:                              0
surf->output->ARH3 format:                              0
surf->output->AMDiS format:                             0

#############################################################################################
#############################################################################################
#\tMESH
surf->space->mesh:\t\t\t\t\tsurfMesh

surfMesh->macro file name:\t\t\t\t./macro/macro.stand.per.2d
surfMesh->periodic file:\t\t\t\t./macro/macro.stand.per

surfMesh->refinement->initial->coarsen allow:         1
surfMesh->refinement->initial->repartition allow:     1
surfMesh->refinement->coarsen allow:                  1
surfMesh->refinement->repartition allow:              0
surfMesh->refinement->initial level:                  2
surfMesh->refinement->global level:                   2

############################################################################################
#\tSOLVER
surf->space->solver:\t\t\t\t\tumfpack
surf->space->solver->ell:                               4
surf->space->solver->max iteration:\t\t\t1000
surf->space->solver->tolerance:\t\t\t\t1.e-8
surf->space->solver->info:\t\t\t\t20
surf->space->solver->left precon:\t\t\tdiag
surf->space->solver->right precon:\t\t\tno

#############################################################################################
surf->space->dim:                                       ${dimension of world}

surf->space->polynomial degree[0]:\t\t\t1
surf->space->polynomial degree[1]:\t\t\t1
surf->space->polynomial degree[2]:\t\t\t1
surf->space->polynomial degree[3]:\t\t\t1

surf->space->marker[0]->strategy:    \t\t\t0  % 0=none, 1=GR, 2=MS, 3=ES, 4=GERS
surf->space->marker[0]->MSGamma:     \t\t\t0.5
surf->space->marker[0]->MSGammaC:    \t\t\t0.1

surf->space->marker[1]->strategy:    \t\t\t0  % 0=none, 1=GR, 2=MS, 3=ES, 4=GERS
surf->space->marker[2]->strategy:    \t\t\t0  % 0=none, 1=GR, 2=MS, 3=ES, 4=GERS
surf->space->marker[3]->strategy:    \t\t\t0  % 0=none, 1=GR, 2=MS, 3=ES, 4=GERS

#############################################################################################
#\tESTIMATOR
surf->space->estimator[0]:\t\t\t\t0
surf->space->estimator[0]->C0:\t\t\t\t1.0
surf->space->estimator[0]->C1:\t\t\t\t1.0
surf->space->estimator[0]->C3:\t\t\t\t1.0

surf->space->estimator[1]:\t\t\t\t0
surf->space->estimator[2]:\t\t\t\t0
surf->space->estimator[3]:\t\t\t\t0

#############################################################################################
#\tADAPT
surf->adapt->timestep:\t\t\t\t\t1.e0
surf->adapt->start time:\t\t\t\t0.0
surf->adapt->end time:\t\t\t\t\t5.

surf->adapt->min timestep:\t\t\t\t1.e-6
surf->adapt->max timestep:\t\t\t\t1.e4

surf->adapt->strategy:\t\t\t\t\t0  % 0=explicit, 1=implicit
surf->adapt->relative energy tolerance:\t\t\t1.e-5 % 0=explicit, 1=implicit
surf->adapt->max iteration:\t\t\t\t100

surf->adapt[0]->tolerance:\t\t\t\t0.05
surf->adapt[0]->time tolerance:\t\t\t\t0.05
surf->adapt[1]->tolerance:\t\t\t\t0.05
surf->adapt[1]->time tolerance:\t\t\t\t0.05
surf->adapt[2]->tolerance:\t\t\t\t0.05
surf->adapt[2]->time tolerance:\t\t\t\t0.05
surf->adapt[3]->tolerance:\t\t\t\t0.05
surf->adapt[3]->time tolerance:\t\t\t\t0.05

WAIT:\t\t\t\t\t\t\t0
"""


def fmt(x):
    s = f"{x:.12g}"
    if "e" not in s and "." not in s:
        s += ".0"
    return s


def build_phi_shape_block(
    n_inner_teeth=3,
    tooth_width=0.12,
    tooth_height=0.22,
    side_width=0.12,
    base_width=1.2,
    base_height=0.2,
    base_center_y=-0.4,
    embed_depth=0.05,
):
    eps = 5.0 / 64.0
    visible_left = -0.5
    visible_right = 0.5
    visible_width = visible_right - visible_left

    if n_inner_teeth < 1:
        raise ValueError("n_inner_teeth deve essere almeno 1.")

    if tooth_width < 2.0 * eps:
        raise ValueError(
            f"tooth_width={tooth_width} troppo piccolo per eps={eps:.6f}; "
            f"usa almeno ~{2*eps:.3f}."
        )

    if embed_depth <= 0:
        raise ValueError("embed_depth deve essere positivo.")

    if embed_depth >= tooth_height:
        raise ValueError("embed_depth deve essere minore di tooth_height.")

    side_height = tooth_height
    inner_available_width = visible_width - 2.0 * side_width
    total_inner_teeth_width = n_inner_teeth * tooth_width

    if total_inner_teeth_width >= inner_available_width:
        raise ValueError(
            "I denti interni occupano tutta la larghezza disponibile; "
            "riduci tooth_width o n_inner_teeth."
        )

    gap = (inner_available_width - total_inner_teeth_width) / (n_inner_teeth + 1)

    base_top = base_center_y + 0.5 * base_height

    tooth_bottom = base_top - embed_depth
    tooth_center_y = tooth_bottom + 0.5 * tooth_height

    side_bottom = base_top - embed_depth
    side_center_y = side_bottom + 0.5 * side_height

    x_start = visible_left + side_width + gap
    inner_centers = [
        x_start + 0.5 * tooth_width + i * (tooth_width + gap)
        for i in range(n_inner_teeth)
    ]

    names = ["rectangle", "rectangle1", "rectangle2"] + [
        f"rectangle{i}" for i in range(3, 3 + n_inner_teeth)
    ]

    lines = [
        "surf->phi->shape:                               " + " + ".join(names),
        "",
        "surf->phi->shape->inner value:                  1",
        "surf->phi->shape->outer value:                  0",
        "surf->phi->shape->center:\t\t        [ 0. , 0. ]",
        "",
        "surf->phi->shape->eps:                          ${surf->eps}",
        "",
        f"rectangle->sides length:\t[{fmt(base_width)},{fmt(base_height)}]",
        f"rectangle->center:\t\t[0.0,{fmt(base_center_y)}]",
        "",
        f"rectangle1->sides length:\t[{fmt(side_width)},{fmt(side_height)}]",
        f"rectangle1->center:\t\t[-0.5,{fmt(side_center_y)}]",
        "",
        f"rectangle2->sides length:\t[{fmt(side_width)},{fmt(side_height)}]",
        f"rectangle2->center:\t\t[0.5,{fmt(side_center_y)}]",
        "",
    ]

    for k, cx in enumerate(inner_centers, start=3):
        lines.append(f"rectangle{k}->sides length:\t[{fmt(tooth_width)},{fmt(tooth_height)}]")
        lines.append(f"rectangle{k}->center:\t\t[{fmt(cx)},{fmt(tooth_center_y)}]")
        if k != 2 + n_inner_teeth:
            lines.append("")

    lines.append("")
    return "\n".join(lines)


def build_pore2d_text(args):
    phi_block = build_phi_shape_block(
        n_inner_teeth=args.n_inner_teeth,
        tooth_width=args.tooth_width,
        tooth_height=args.tooth_height,
        side_width=args.side_width,
        base_width=args.base_width,
        base_height=args.base_height,
        base_center_y=args.base_center_y,
        embed_depth=args.embed_depth,
    )
    return HEADER + "\n" + phi_block + FOOTER


def main():
    parser = argparse.ArgumentParser(
        description="Genera pore2D.dat con base inferiore, lati verticali e denti interni."
    )
    parser.add_argument("-o", "--output", default="pore2D.dat")
    parser.add_argument("--n-inner-teeth", type=int, default=3)
    parser.add_argument("--tooth-width", type=float, default=0.16)
    parser.add_argument("--tooth-height", type=float, default=0.52)
    parser.add_argument("--side-width", type=float, default=0.1)
    parser.add_argument("--base-width", type=float, default=1.2)
    parser.add_argument("--base-height", type=float, default=0.3)
    parser.add_argument("--base-center-y", type=float, default=-0.4)
    parser.add_argument("--embed-depth", type=float, default=0.05)

    args = parser.parse_args()
    text = build_pore2d_text(args)
    Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()