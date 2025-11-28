import sys

# 一つ下のディレクトリにあるモジュールもimportできるようにする。
sys.path.insert(0, "..")

# commonはいずれ独立したmoduleにする。
from common import gromacs2
import numpy as np

kB = 1.380649e-23  # Boltzmann constant
NA = 6.02214076e23  # Avogadro constant
q = 1.602176634e-19  # Unit charge
CC = 8.987552e9  # Coulomb constant


def interactions_tip4pice(
    atom_pos: np.ndarray, target: int, com: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """TIP4P/Iceモデルの1分子と残りすべてとの相互作用

    Args:
        atom_pos (np.ndarray): 三次元配列 (分子番号 x 原子 x 空間次元)
        target (int): 注目する分子の番号
        com (np.ndarray): 重心位置(分子番号x空間次元)
        cell (np.ndarray): セル行列(空間次元x空間次元)

    Returns:
        np.ndarray: 相互作用エネルギー J/mol 自分自身との相互作用は0とする。
    """
    # http://www.sklogwiki.org/SklogWiki/index.php/TIP4P/Ice_model_of_water
    qH = 0.5897 * q
    qM = -qH * 2
    eps_oo = 106.1 * kB * NA  # J/mol
    sig_oo = 0.31668  # nm
    econst = NA * CC / 1e-9

    celli = np.linalg.inv(cell)
    Nmol = atom_pos.shape[0]

    relpos = com - com[target]
    # 相互作用相手の分子のセル
    cell_offset = np.floor(relpos @ celli + 0.5) @ cell

    # 重心の相対位置を、PBCに従い調整
    relpos -= cell_offset

    # 重心間距離が0.9 nmより近いならTrue
    prox = np.sum(relpos * relpos, axis=1) < 0.9**2

    # 原子対ごとに距離を求め、エネルギーを計算する
    interactions = np.zeros(Nmol)

    mm = atom_pos[:, 3] - atom_pos[target, 3] - cell_offset
    interactions += econst * qM * qM / np.linalg.norm(mm, axis=1)

    for j in (1, 2):
        for k in (1, 2):
            hh = atom_pos[:, j] - atom_pos[target, k] - cell_offset
            interactions += econst * qH * qH / np.linalg.norm(hh, axis=1)

    for j in (1, 2):
        mh = atom_pos[:, j] - atom_pos[target, 3] - cell_offset
        interactions += econst * qH * qM / np.linalg.norm(mh, axis=1)
        mh = atom_pos[:, 3] - atom_pos[target, j] - cell_offset
        interactions += econst * qH * qM / np.linalg.norm(mh, axis=1)

    oo = atom_pos[:, 0] - atom_pos[target, 0] - cell_offset
    # 酸素酸素間距離の配列
    r_oo = np.linalg.norm(oo, axis=1)
    interactions += 4 * eps_oo * ((sig_oo / r_oo) ** 12 - (sig_oo / r_oo) ** 6)

    # proxをかけることで、0.9 nmよりも重心間距離が遠い対は相互作用しない。
    interactions *= prox

    # 自分自身との相互作用は除外する
    interactions[target] = 0

    # 相互作用を返す。
    return interactions


def main():
    for frame in gromacs2.read_gro(sys.stdin):
        # 分子ごとにきりわける
        mols = frame.decompose()
        waters = mols["water"]
        if not waters:
            waters = mols["SOL"]
        if not waters:
            waters = mols["ICE"]

        # 出力は、グラフを作るときにやりやすいように、水分子の位置と、周囲との相互作用だけにする。
        atom_names = waters.atoms
        Nmol = waters.positions.shape[0]

        # 水分子がPBCでばらけているやつがいるらしい。
        # 酸素との相対位置になおし、修正する。
        celli = np.linalg.inv(frame.cell)
        for i in (1, 2, 3):
            waters.positions[:, i] -= waters.positions[:, 0]
        waters.positions[:, 1:4] -= (
            np.floor(waters.positions[:, 1:4] @ celli + 0.5) @ frame.cell
        )
        for i in (1, 2, 3):
            waters.positions[:, i] += waters.positions[:, 0]

        # 念のため、OHが離れすぎている場合は打ち切る
        assert np.all(
            np.sum((waters.positions[:, 0] - waters.positions[:, 1]) ** 2, axis=1)
            < 0.01
        )
        assert np.all(
            np.sum((waters.positions[:, 0] - waters.positions[:, 2]) ** 2, axis=1)
            < 0.01
        )

        # まず、周期境界条件のために、分子の重心Center of Massを計算しておく
        com = (
            waters.positions[:, 0] * 16
            + waters.positions[:, 1]
            + waters.positions[:, 2]
        ) / 18

        for i in range(Nmol):
            interactions = interactions_tip4pice(waters.positions, i, com, frame.cell)
            x, y, z = com[i]
            print(f"{x:.4f} {y:.4f} {z:.4f} {np.sum(interactions)/1000:.4f}")
        # 空行で仕切る
        print()


if __name__ == "__main__":
    main()
