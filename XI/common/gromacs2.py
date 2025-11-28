#!/usr/bin/env python

import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Iterable, Dict, Union
import numpy as np
from logging import getLogger


@dataclass
class Residue:
    positions: Iterable
    resname: str = ""
    atoms: Tuple = ""


@dataclass
class Frame:
    residue_id: Iterable
    residue_name: Iterable
    atom_id: Iterable
    atom_name: Iterable
    position: Iterable
    cell: Iterable

    def write_gro(self, file, remark="Written by write_gro"):
        """
        fileにframeを書きだす。
        """
        logger = getLogger()

        # 1行目はメッセージ行
        print(remark, file=file)
        # 2行目は原子数
        Natom = len(self.position)
        print(Natom, file=file)
        logger.debug(self.residue_id.shape)
        # 原子もそのまま
        for i in range(Natom):
            ri = self.residue_id[i]
            r = self.residue_name[i]
            a = self.atom_name[i]
            ai = self.atom_id[i]
            pos = self.position[i]
            logger.debug((ri, r, a, ai, pos))
            print(
                f"{ri:5d}{r:5s}{a:>5s}{ai:5d}{pos[0]:8.3f}"
                f"{pos[1]:8.3f}{pos[2]:8.3f}",
                file=file,
            )
        # セルは、直方体とそれ以外で書き方が違う
        cell = self.cell
        if cell[1, 0] == 0:
            print(cell[0, 0], cell[1, 1], cell[2, 2], file=file)
        else:
            print(
                cell[0, 0],
                cell[1, 1],
                cell[2, 2],
                cell[1, 0],
                cell[2, 0],
                cell[0, 1],
                cell[2, 1],
                cell[0, 2],
                cell[1, 2],
                file=file,
            )

    def decompose(self) -> Dict:
        """read_gro3で読みこんだ原子列を、分子単位に切りわける。"""
        Natom = self.position.shape[0]
        molecules = defaultdict(dict)
        last_residue_id = -1
        atom_names = []
        atom_positions = []
        for i in range(Natom):
            residue_id = self.residue_id[i]
            if len(atom_names) > 0 and residue_id != last_residue_id:
                atom_names = tuple(atom_names)
                # 既出のresnameの場合は
                if residue_name in molecules:
                    assert molecules[residue_name].atoms == atom_names
                else:
                    molecules[residue_name] = Residue(atoms=atom_names, positions=[])
                molecules[residue_name].positions.append(atom_positions)
                atom_names = []
                atom_positions = []
            last_residue_id = residue_id
            residue_name = self.residue_name[i]
            atom_name = self.atom_name[i]
            atom_position = self.position[i].copy()

            atom_names.append(atom_name)
            atom_positions.append(atom_position)

        if len(atom_names) > 0:
            atom_names = tuple(atom_names)
            # 既出のresnameの場合は
            if residue_name in molecules:
                assert molecules[residue_name].atoms == atom_names
            else:
                molecules[residue_name] = Residue(atoms=atom_names, positions=[])
            molecules[residue_name].positions.append(atom_positions)

        for residue_name, residue in molecules.items():
            residue.positions = np.array(residue.positions)
        return molecules

    def append(self, frame, new_cell: Union[np.ndarray, None] = None):
        if new_cell is None:
            new_cell = self.cell
        self.atom_id = np.concatenate([self.atom_id, frame.atom_id], axis=0)
        self.position = np.concatenate([self.position, frame.position], axis=0)
        self.atom_name = np.concatenate([self.atom_name, frame.atom_name], axis=0)
        self.residue_id = np.concatenate([self.residue_id, frame.residue_id], axis=0)
        self.residue_name = np.concatenate(
            [self.residue_name, frame.residue_name], axis=0
        )
        self.cell = new_cell


def read_gro(file):
    """
    gromacsの.groファイルを読みこむ。

    あとで出力する場合にそなえ、できるだけデータをそのままの形で保持する。
    """

    # 無限ループ
    while True:
        frame = Frame(
            residue_id=[],
            residue_name=[],
            atom_id=[],
            atom_name=[],
            position=[],
            cell=None,
        )

        title = file.readline()
        # 終了判定。1文字も読めない時はファイルの終わり。
        if len(title) == 0:
            return
        n_atom = int(file.readline())
        for i in range(n_atom):
            line = file.readline()
            residue_id = int(line[0:5])
            residue = line[5:10].strip()
            atom = line[10:15].strip()
            atom_id = int(line[15:20])
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
            # 速度は省略

            frame.residue_id.append(residue_id)
            frame.residue_name.append(residue)
            frame.atom_name.append(atom)
            frame.atom_id.append(atom_id)
            frame.position.append([x, y, z])

        cell = [float(x) for x in file.readline().split()]

        # numpy形式に変換しておく。
        frame.residue_id = np.array(frame.residue_id)
        frame.residue_name = np.array(frame.residue_name)
        frame.atom_name = np.array(frame.atom_name)
        frame.atom_id = np.array(frame.atom_id)
        frame.position = np.array(frame.position)

        # cellは行列の形にしておく。
        if len(cell) == 3:
            # 直方体セルの場合
            cell = np.diag(cell)
        else:
            # 9パラメータで指定される場合は、順番がややこしい。
            # v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
            x = [cell[0], cell[5], cell[7]]
            y = [cell[3], cell[1], cell[8]]
            z = [cell[4], cell[6], cell[2]]
            cell = np.array([x, y, z])

        frame.cell = cell
        # returnの代わりにyieldを使うと、繰り返し(iterator)にできる。
        yield frame


# def compose(mols, cell):
#     resi_id = []
#     residue = []
#     atom = []
#     atom_id = []
#     position = []
#     aid = 0
#     rid = 0
#     for name, mollist in mols.items():
#         for mol in mollist:
#             rid += 1
#             for a in mol:
#                 aid += 1
#                 resi_id.append(rid)
#                 residue.append(name)
#                 atom.append(a[0])
#                 atom_id.append(aid)
#                 position.append(a[1])
#     return {
#         "resi_id": resi_id,
#         "residue": residue,
#         "atom": atom,
#         "atom_id": atom_id,
#         "position": position,
#         "cell": cell,
#     }


if __name__ == "__main__":
    for frame in read_gro(sys.stdin):
        print(frame)
