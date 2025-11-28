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
        HB = []
        for o, h, distance in pairs_iter(rel_O, maxdist=0.25, cell=cell, pos2=rel_H):
            if h // 2 == o:
                # in the same molecule; covalent bond
                continue
            HB.append((o, h // 2))

        G = nx.Graph(HB)
        bin_width = 0.5  # nm
        zbins = [
            {4: 0, 5: 0, 6: 0, 7: 0} for i in range(int(cell[2, 2] / bin_width) + 1)
        ]
        zticks = np.arange(0, cell[2, 2], bin_width)
        if True:  # for debug
            for cycle in cycles.cycles_iter(G, maxsize=7, pos=rel_O):
                cycle_size = len(cycle)
                g = nx.subgraph(G, cycle)
                center = cycles.centerOfMass(cycle, rel_O) @ cell
                bin = int(center[2] / bin_width)
                if 4 <= cycle_size <= 7:
                    zbins[bin][cycle_size] += 1

            # 各binの合計を計算して比率に変換
            totals = [bin[4] + bin[5] + bin[6] + bin[7] for bin in zbins]
            ratios_4 = [
                bin[4] / total if total > 0 else 0 for bin, total in zip(zbins, totals)
            ]
            ratios_5 = [
                bin[5] / total if total > 0 else 0 for bin, total in zip(zbins, totals)
            ]
            ratios_6 = [
                bin[6] / total if total > 0 else 0 for bin, total in zip(zbins, totals)
            ]
            ratios_7 = [
                bin[7] / total if total > 0 else 0 for bin, total in zip(zbins, totals)
            ]

            plt.bar(zticks, ratios_4, width=bin_width, label="4-membered")
            plt.bar(
                zticks,
                ratios_5,
                width=bin_width,
                label="5-membered",
                bottom=ratios_4,
            )
            plt.bar(
                zticks,
                ratios_6,
                width=bin_width,
                label="6-membered",
                bottom=[r4 + r5 for r4, r5 in zip(ratios_4, ratios_5)],
            )
            plt.bar(
                zticks,
                ratios_7,
                width=bin_width,
                label="7-membered",
                bottom=[
                    r4 + r5 + r6 for r4, r5, r6 in zip(ratios_4, ratios_5, ratios_6)
                ],
            )
            plt.xlabel("z (nm)")
            plt.ylabel("ratio")
            plt.title("Ratio of cycles per z-slice")
            plt.legend()
            plt.savefig("cycles.pdf")
            plt.savefig("cycles.png")
            plt.show()

        DG = nx.DiGraph(HB)
        zbins = [defaultdict(int) for i in range(int(cell[2, 2] / bin_width) + 1)]
        for ring in rings.cycle_orientations_iter(DG, maxsize=6, pos=rel_O):
            cycle_size = len(ring.path)
            if cycle_size != 6:
                continue
            center = cycles.centerOfMass(ring.path, rel_O) @ cell
            bin = int(center[2] / bin_width)
            zbins[bin][ring.code] += 1

        for i, bin in enumerate(zbins):
            for code in bin:
                print(code, bin[code])

        # 各binの合計を計算して比率に変換
        codes = [0, 1, 3, 5, 7, 9, 11, 21]
        totals = [sum(bin[code] for code in codes) for bin in zbins]
        ratios = {}
        for code in codes:
            ratios[code] = [
                bin[code] / total if total > 0 else 0
                for bin, total in zip(zbins, totals)
            ]

        plt.bar(zticks, ratios[0], width=bin_width, label="0")
        bottom = ratios[0]
        plt.bar(zticks, ratios[1], width=bin_width, label="1", bottom=bottom)
        bottom = [bottom[i] + ratios[1][i] for i in range(len(bottom))]
        plt.bar(zticks, ratios[3], width=bin_width, label="3", bottom=bottom)
        bottom = [bottom[i] + ratios[3][i] for i in range(len(bottom))]
        plt.bar(zticks, ratios[5], width=bin_width, label="5", bottom=bottom)
        bottom = [bottom[i] + ratios[5][i] for i in range(len(bottom))]
        plt.bar(zticks, ratios[7], width=bin_width, label="7", bottom=bottom)
        bottom = [bottom[i] + ratios[7][i] for i in range(len(bottom))]
        plt.bar(zticks, ratios[9], width=bin_width, label="9", bottom=bottom)
        bottom = [bottom[i] + ratios[9][i] for i in range(len(bottom))]
        plt.bar(
            zticks,
            ratios[11],
            width=bin_width,
            label="11",
            bottom=bottom,
        )
        bottom = [bottom[i] + ratios[11][i] for i in range(len(bottom))]
        plt.bar(
            zticks,
            ratios[21],
            width=bin_width,
            label="21",
            bottom=bottom,
        )
        plt.xlabel("z (nm)")
        plt.ylabel("ratio")
        plt.title("Ratio of cycles per z-slice")
        plt.legend()
        plt.savefig("rings.pdf")
        plt.savefig("rings.png")
        plt.show()
