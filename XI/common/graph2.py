import numpy as np
import pairlist as pl
from logging import getLogger
import networkx as nx
import json
from dataclasses import dataclass

# for gromacs2.py's Frame object


# copied from ../Solution/cycles.py
def gro2atoms(frame, O="OW", H="HW"):
    """groファイルから水の原子位置を割り出して相対座標を返す

    Args:
        frame (_type_): _description_
        O (str, optional): _description_. Defaults to "OW".
        H (str, optional): _description_. Defaults to "HW".

    Returns:
        _type_: _description_
    """
    cell = frame.cell
    celli = np.linalg.inv(cell)
    oxygens = np.array(
        [
            frame.position[i]
            for i in range(len(frame.atom_name))
            if frame.atom_name[i][: len(O)] == O
        ]
    )
    hydrogens = np.array(
        [
            frame.position[i]
            for i in range(len(frame.atom_name))
            if frame.atom_name[i][: len(H)] == H
        ]
    )

    # in a fractional coordinate
    o_frac = oxygens @ celli
    h_frac = hydrogens @ celli
    return o_frac, h_frac, cell


def OH2graph(o_frac, h_frac, cell):
    """酸素と水素の位置から水素結合ネットワークを割りだす。

    Args:
        o_frac (_type_): _description_
        h_frac (_type_): _description_
        cell (_type_): _description_

    Returns:
        _type_: _description_
    """
    HBs = dict()
    # 酸素と水素の距離が0.25 nm以下の組みあわせをさがし、
    for i, j, d in pl.pairs_iter(o_frac, 0.25, cell, pos2=h_frac):
        # 同じ分子内の酸素と水素でなければ、
        if d > 0.1:
            # 水素の番号を分子の番号に換算
            jo = j // 2
            # すでに水素結合がある場合は
            if (jo, i) in HBs:
                # 新しいほうが長ければ
                if HBs[jo, i] < d:
                    # 見送る
                    continue
            # 水素結合を登録
            # 分岐水素結合もできるかもしれないが気にしない。
            HBs[jo, i] = d
    return nx.Graph(HBs.keys())


def center_of_mass(g: nx.Graph, frac, cell):
    """グラフの重心座標

    Args:
        g (nx.Graph): 部分グラフ
        frac (_type_): (部分グラフではなくもとの全グラフの)すべてのノードのセル相対位置
        cell (_type_): セル行列
    """
    # 最初のノードから見た、他のノードの相対位置
    nodes = list(g.nodes)
    first = nodes[0]
    x = frac[nodes] - frac[first]
    # 相対位置に周期境界条件を施す
    x -= np.floor(x + 0.5)
    # 重心
    com = np.mean(x, axis=0)
    return com + frac[first]


def center_graph(g: nx.Graph, frac, cell):
    """グラフの重心座標を原点とした、ノードの位置

    Args:
        g (nx.Graph): 部分グラフ
        frac (_type_): (部分グラフではなくもとの全グラフの)すべてのノードのセル相対位置
        cell (_type_): セル行列
    """
    # 最初のノードから見た、他のノードの相対位置
    nodes = list(g.nodes)
    first = nodes[0]
    x = frac[nodes] - frac[first]
    # 相対位置に周期境界条件を施す
    x -= np.floor(x + 0.5)
    # 重心
    com = np.mean(x, axis=0)
    # 重心を原点とし、絶対座標になおす
    abso = (x - com) @ cell
    return {int(node): x for node, x in zip(nodes, abso)}


def serialize(g: nx.Graph) -> str:
    """グラフを文字列で表現する

    Args:
        g (nx.Graph): the undirected graph to be encoded.

    Returns:
        str: a string representing the graph.
    """

    return ",".join(f"{x}-{y}" for x, y in g.edges())


def deserialize(s):
    """文字列からグラフに復元

    Args:
        s (_type_): _description_

    Returns:
        _type_: _description_
    """
    edges = [e.split("-") for e in s.split(",")]
    return nx.Graph(edges)


@dataclass
class Graph3D:
    """ノードの絶対座標つきグラフ。

    グラフのノード名は(JSONの仕様にあわせ)数字ではなく文字列とする。
    """

    graph: nx.Graph
    position: dict

    def load(self, id):
        filename = f"ref/{id}.json"
        with open(filename) as f:
            data = json.load(f)
        self.graph = deserialize(data["graph"])
        self.position = {k: np.array(v) for k, v in data["nodes"].items()}

    def dump(self, id, **kwarg):
        filename = f"ref/{id}.json"
        with open(filename, "w") as filehandle:
            json.dump(
                dict(
                    graph=serialize(self.graph),
                    nodes={k: list(v) for k, v in self.position.items()},
                ),
                filehandle,
                **kwarg,
            )
