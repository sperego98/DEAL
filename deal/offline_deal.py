from __future__ import annotations

from dataclasses import dataclass,field,asdict
from typing import List, Optional, Sequence
from typing import List, Dict, Any

from copy import deepcopy

import numpy as np
from ase.io import iread, write
from ase.calculators.singlepoint import SinglePointCalculator

from flare.learners.utils import is_std_in_bound
from flare.atoms import FLARE_Atoms

from deal import get_sgp_calc
from utils import create_chemiscope_input

import argparse
import sys
import yaml
from pprint import pformat

@dataclass
class DataConfig:
    # --- data / trajectory ---
    files: List[str]
    format: Optional[str] = None
    index: str = ":"                 # ASE selection string
    # TODO ADD SHUFFLE

@dataclass
class DEALConfig:
    # --- selection parameters ---
    threshold: float = 1.0
    update_threshold: Optional[float] = None
    max_atoms_added: int = -1
    min_steps_with_model: int = 0     # frames between two selections

    # --- GP training options ---
    force_only: bool = True           # ignore energies/stress if True
    train_hyps: bool = False          # train hyperparams after each update

    # --- output ---
    output_prefix: str = "deal"
    verbose: bool = False

@dataclass
class FlareConfig:
    # --- gp ---
    gp: str = "SGP_Wrapper"

    # --- kernel ---
    kernels: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "NormalizedDotProduct", "sigma": 2.0, "power": 2}
    ])
    # --- descriptor ---
    descriptors: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "name": "B2",
            "nmax": 8,
            "lmax": 3,
            "cutoff_function": "cosine",
            "radial_basis": "chebyshev",
        }
    ])
    # --- species ---
    species: list[int] = None 
    # --- parameters ---
    cutoff: float = 4.5
    variance_type: str = "local"
    max_iterations: int = 20

class DEAL:
    """
    Minimal DEAL-style selector:

    - iterate over a trajectory (ASE files)
    - for each frame:
        * read energy/forces/stress from the file
        * evaluate SGP and its uncertainties
        * if uncertainty is above threshold -> "select" frame and update SGP

    Outputs:
        <output_prefix>_selected.xyz
        <output_prefix>_flare.json
        <output_prefix>_chemiscope.json.gz
    """

    def __init__(self, 
                 data_cfg: DataConfig, 
                 deal_cfg: DEALConfig, 
                 flare_cfg: FlareConfig):
        
        self.data_cfg = data_cfg
        self.deal_cfg = deal_cfg
        self.flare_cfg = flare_cfg

        # Automatically detect species if not specified
        if self.flare_cfg.species is None:
            self.flare_cfg.species = self._get_species()

        # Print configuratons is requested:
        if self.deal_cfg.verbose:

            print('[INFO] Configurations:')
            print('-',pformat(self.data_cfg))
            print('-',pformat(self.deal_cfg))
            print('-',pformat(self.flare_cfg))
            print('')

        # Build SGP calculator using your existing helper
        self.flare_calc, self.kernels = get_sgp_calc(asdict(self.flare_cfg))
        self.gp = self.flare_calc.gp_model

        # Default update_threshold if not set
        if self.deal_cfg.update_threshold is None:
            self.deal_cfg.update_threshold = 0.8 * self.deal_cfg.threshold

        self.selected_frames: List = []
        self.dft_count: int = 0
        self.last_dft_step: int = -10**9   # effectively -∞

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------

    def _frames(self):
        """Generator over all frames in all trajectory files."""
        for fname in self.data_cfg.files:
            for at in iread(fname, index=self.data_cfg.index, format=self.data_cfg.format):
                yield at

    def _get_species(self):
        """
        Detect species automatically using the DataConfig instance.
        Reads only the first frame from the first file.
        """

        for fname in self.data_cfg.files:
            for atoms in iread(fname, index=self.data_cfg.index, format=self.data_cfg.format):
                return sorted(set(atoms.get_atomic_numbers().tolist()))

    def _extract_dft(self, ase_atoms):
        """
        Extract DFT forces / energy / stress from a frame.

        Assumes the extxyz was written with energies and forces and that
        ASE has attached a SinglePointCalculator to atoms.calc.
        """
        if ase_atoms.calc is None:
            raise RuntimeError(
                "Frame has no calculator attached. Make sure your extxyz "
                "contains energies/forces so ASE builds a SinglePointCalculator."
            )

        res = ase_atoms.calc.results
        forces = np.array(res["forces"])
        energy = float(res.get("energy", ase_atoms.info.get("energy", 0.0)))
        stress = res.get("stress", None)

        return forces, energy, stress

    def _print_progress(self, step: int):
        msg = f"[DEAL] Examined: {step+1} | Selected: {self.dft_count}"
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()


    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        for step, ase_frame in enumerate(self._frames()):
            # 1) DFT labels from original ASE frame
            dft_forces, dft_energy, dft_stress = self._extract_dft(ase_frame)

            # 2) Convert to FLARE_Atoms for SGP calculations & uncertainties
            atoms = FLARE_Atoms.from_ase_atoms(ase_frame)

            # 2a) INITIALIZATION: if GP has no training data, use first frame
            #     to bootstrap the model (no uncertainty check).
            if len(self.gp.training_data) == 0:
                self._update_gp(
                    atoms=atoms,
                    train_atoms=list(range(len(atoms))),
                    dft_frcs=dft_forces,
                    dft_energy=dft_energy,
                    dft_stress=dft_stress,
                )
                self.last_dft_step = step
                self._store_selected_frame(step, ase_frame,
                                           target_atoms=list(range(len(atoms))))
                continue

            # 3) Predict with SGP and compute uncertainties
            atoms.calc = self.flare_calc
            _ = atoms.get_forces()   # triggers GP eval and stores stds internally

            std_in_bound, target_atoms = is_std_in_bound(
                self.deal_cfg.threshold,
                self.gp.force_noise,
                atoms,
                max_atoms_added=self.deal_cfg.max_atoms_added,
                update_style="add_n",
                update_threshold=self.deal_cfg.update_threshold,
            )

            steps_since_last = step - self.last_dft_step
            if (not std_in_bound) and (steps_since_last >= self.deal_cfg.min_steps_with_model):
                # Select this frame & update GP
                self.last_dft_step = step
                self._store_selected_frame(step, ase_frame, target_atoms)
                self._update_gp(
                    atoms=atoms,
                    train_atoms=list(target_atoms),
                    dft_frcs=dft_forces,
                    dft_energy=dft_energy,
                    dft_stress=dft_stress,
                )
            
            # ========== print progress ==========
            self._print_progress(step)

        # newline so terminal prompt doesn't collide with progress line
        print('\n')

        # ------------------------------------------------------------------
        # outputs
        # ------------------------------------------------------------------
        if self.selected_frames:
            out_xyz = f"{self.deal_cfg.output_prefix}_selected.xyz"
            write(out_xyz, self.selected_frames)
            try:
                create_chemiscope_input(
                    trajectory=out_xyz,
                    filename=f"{self.deal_cfg.output_prefix}_chemiscope.json.gz",
                )
            except Exception as exc:
                print(f"[DEAL] Could not write chemiscope file: {exc}")

        # Save final SGP model
        self.flare_calc.write_model(f"{self.deal_cfg.output_prefix}_flare.json")

    # ------------------------------------------------------------------
    # bookkeeping & GP update (ported from OTF_DEAL.update_gp)
    # ------------------------------------------------------------------

    def _store_selected_frame(self, step: int, ase_frame, target_atoms: Sequence[int]):
        """Keep a copy of the selected ASE frame for writing to XYZ."""
        sel = ase_frame.copy()
        sel.info["step"] = step
        sel.info["target_atoms"] = np.array(target_atoms, dtype=int)
        self.selected_frames.append(sel)
        self.dft_count += 1

    def _update_gp(
        self,
        atoms,
        train_atoms: Sequence[int],
        dft_frcs: np.ndarray,
        dft_energy: float | None = None,
        dft_stress: np.ndarray | None = None,
    ) -> None:
        """
        Update the SGP using DFT forces (and optionally energies/stress)
        on the current FLARE_Atoms structure.

        This mirrors your original OTF_DEAL.update_gp, but without
        wall-time logging or mapping.
        """
        # Convert stress into FLARE convention if present
        flare_stress = None
        if dft_stress is not None:
            dft_stress = np.asarray(dft_stress)
            # allow either 3x3 tensor or 6-vector from ASE
            if dft_stress.shape == (3, 3):
                xx, yy, zz = dft_stress[0, 0], dft_stress[1, 1], dft_stress[2, 2]
                yz, xz, xy = dft_stress[1, 2], dft_stress[0, 2], dft_stress[0, 1]
                dft_stress_voigt = np.array([xx, yy, zz, yz, xz, xy])
            else:
                dft_stress_voigt = dft_stress

            # ASE uses +sigma; FLARE uses -sigma in this convention
            flare_stress = -np.array(
                [
                    dft_stress_voigt[0],
                    dft_stress_voigt[5],
                    dft_stress_voigt[4],
                    dft_stress_voigt[1],
                    dft_stress_voigt[3],
                    dft_stress_voigt[2],
                ]
            )

        if self.deal_cfg.force_only:
            dft_energy = None
            flare_stress = None

        # Store a copy of the structure, attach DFT labels via SinglePointCalculator
        struc_to_add = deepcopy(atoms)

        sp_results = {"forces": dft_frcs}
        if dft_energy is not None:
            sp_results["energy"] = dft_energy
        if flare_stress is not None:
            sp_results["stress"] = flare_stress

        struc_to_add.calc = SinglePointCalculator(struc_to_add, **sp_results)

        # Update GP database (this is where FLARE C++ gets called)
        self.gp.update_db(
            struc_to_add,
            dft_frcs,
            custom_range=list(train_atoms),
            energy=dft_energy,
            stress=flare_stress,
        )

        # Update internal L and alpha
        self.gp.set_L_alpha()

        # Optionally train hyperparameters
        if self.deal_cfg.train_hyps:
            self.gp.train(logger_name=None)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DEAL selector: read a YAML config or specify a trajectory and (optionally) a threshold."
    )

    parser.add_argument(
        "-c", "--config",
        dest="config",
        help="YAML configuration file."
    )
    parser.add_argument(
        "-f", "--file",
        dest="filename",
        help="Input trajectory file (e.g. traj.xyz)."
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        dest="threshold",
        help="GP uncertainty threshold that triggers selection."
    )

    return parser.parse_args()


def main() -> None:

    print("""
888888ba   88888888b  .d888888  88        
88     8b  88        d8     88  88        
88     88  88aaaa    88aaaaa88  88        
88     88  88        88     88  88        
88     8P  88        88     88  88        
8888888P   88888888P 88     88  888888888
""")

    args = parse_args()
    cfg_dict = {}

    # Start from YAML config if provided
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}

    # Initialize dicts if not available
    for key in ["data", "deal", "flare"]:
        if key not in cfg_dict:
            cfg_dict[key] = {}

    # Overwrite / fill from CLI options
    if args.filename is not None:
        cfg_dict["data"]["files"] = [args.filename]
    if args.threshold is not None:
        cfg_dict["deal"]["threshold"] = args.threshold

    # Check file
    try:
        cfg_dict["data"]["files"][0]
    except KeyError:
        print('[ERROR] No input trajectory specified. Please provide a trajectory file (-f/--file) or a YAML config file (-c/--config).')
        sys.exit(1)

    # Build configs and run
    data_cfg = DataConfig(**cfg_dict["data"])
    deal_cfg = DEALConfig(**cfg_dict["deal"])
    flare_cfg = FlareConfig(**cfg_dict["flare"])

    # Handle multiple thresholds
    if isinstance(deal_cfg.threshold, list):
        prefix = deal_cfg.output_prefix
        for threshold in deal_cfg.threshold:
            print('[INFO] Running DEAL with threshold', threshold)
            deal_cfg.threshold = threshold
            deal_cfg.output_prefix = f"{prefix}_{str(threshold).replace('-','')}"
            DEAL(data_cfg, deal_cfg, flare_cfg).run()
    else:
        DEAL(data_cfg, deal_cfg, flare_cfg).run()

if __name__ == "__main__":
    main()

