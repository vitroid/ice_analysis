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

        # yaplotの1フレームを開始。矢印の表現を指定。
        s = yap.ArrowType(2)
        # 矢印の幅を指定
        s += yap.Size(0.05)
        # 六角形の輪を一つずつ処理。
        cnt = 0
        for ring in rings.cycle_orientations_iter(DG, maxsize=6, pos=rel_O):
            cnt += 1
            cycle_size = len(ring.path)
            if cycle_size != 6:
                continue
            # if cnt != 100:
            #     continue
            # 有向グラフ上の六角形に関して、重心座標を絶対座標系で求める。
            center = cycles.centerOfMass(ring.path, rel_O) @ cell

            # 六角形に沿ったベクトルの輪。分極した輪では0にならない。
            net_dipole = np.zeros(3)
            cyc = list(ring.path) + [ring.path[0]]
            logger.debug(ring)
            for k, (i, j) in enumerate(zip(cyc[0:], cyc[1:])):
                dipole = rel_O[j] - rel_O[i]
                dipole -= np.floor(dipole + 0.5)
                if ring.ori[k]:
                    pass
                    # s += yap.Arrow(rel_O[i] @ cell, (rel_O[i] + dipole) @ cell)
                else:
                    dipole = -dipole
                    # s += yap.Arrow(rel_O[j] @ cell, (rel_O[j] + dipole) @ cell)
                net_dipole += dipole
                logger.debug(dipole)
            net_dipole = net_dipole @ cell * 0.15
            logger.debug(net_dipole)
            s += yap.Arrow(center - net_dipole, center + net_dipole)

        print(s + yap.NewPage())
