"""Convert SiO2 LAMMPS atomic data to ExtXYZ (type 1=O, type 2=Si)."""

import argparse
from pathlib import Path

from ase.io import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path,
        default=Path(__file__).parent / "random_sio2_size_300_demo.dat",
        help="Input file; defaults to the 300-atom example in this script's directory",
    )
    parser.add_argument("--output", type=Path, help="Output path; defaults to the input path with an .extxyz suffix")
    args = parser.parse_args()
    output = args.output or args.input.with_suffix(".extxyz")

    if output.exists():
        parser.error(f"Output file already exists: {output}. Use --output to specify another filename.")

    # LAMMPS type IDs are not atomic numbers; explicitly map O=8 and Si=14.
    atoms = read(
        args.input,
        format="lammps-data",
        atom_style="atomic",
        Z_of_type={1: 8, 2: 14},
        units="metal",
    )
    write(output, atoms, format="extxyz")
    print(f"Converted {len(atoms)} atoms: {output}")


if __name__ == "__main__":
    main()
