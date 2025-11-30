# cyclessを使い、6員環の矢印の向きの統計をzスライスごとにとる。

from cycless import cycles, dicycles, rings
import networkx as nx
from collections import defaultdict

# .groを読みこむ

# commonはひとつ下のディレクトリにある。

from common.gromacs2 import read_gro
import numpy as np
from pairlist import pairs_iter
import matplotlib.pyplot as plt
import sys
import yaplotlib as yap
from logging import getLogger, DEBUG, INFO
import logging

logging.basicConfig(level=INFO)
logger = getLogger(__name__)

if len(sys.argv) > 1:
    gro_file = sys.argv[1]
else:
    gro_file = "00400.40.gro"

with open(gro_file, "r") as f:
    for frame in read_gro(f):
        molecules = frame.decompose()
        # 原子の座標
        O = molecules["water"].positions[:, 0]
        H = molecules["water"].positions[:, 1:3].reshape(-1, 3)

        # セル行列とその逆行列
        cell = frame.cell
        celli = np.linalg.inv(cell)

        # セル相対座標に変換
        rel_O = O @ celli
        rel_H = H @ celli

        # 水素結合ネットワークを再構成
        DG = nx.DiGraph(
            [
                [o, h // 2]
                for o, h, _ in pairs_iter(rel_O, maxdist=0.25, cell=cell, pos2=rel_H)
                if h // 2 != o
            ]
        )

        grids = defaultdict(list)
        grid_size = 0.7  # nm

        for edge in DG.edges:
            o, h = edge
            delta = rel_O[h] - rel_O[o]
            delta -= np.floor(delta + 0.5)
            center = rel_O[o] + delta / 2
            # [0..1)
            center -= np.floor(center)
            grid = center @ cell / grid_size
            grid = tuple((int(x) for x in grid))
            grids[grid].append(delta @ cell)

        # yaplotの1フレームを開始。矢印の表現を指定。
        s = yap.ArrowType(2)
        # 矢印の幅を指定
        s += yap.Size(0.05)
        for grid, dipoles in grids.items():
            dipole = np.mean(dipoles, axis=0) * 3
            center = np.array(grid) * grid_size
            s += yap.Arrow(center - dipole, center + dipole)

        print(s + yap.NewPage())
