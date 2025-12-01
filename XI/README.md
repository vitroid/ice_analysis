水素結合の秩序を解析します。

# セットアップ手順

## 1. リポジトリのクローン

```shell
git clone https://github.com/vitroid/ice_analysis
cd XI
```

## 2. Poetry のインストール（未インストールの場合）

Poetry がインストールされていない場合は、以下のコマンドでインストールします：

```shell
curl -sSL https://install.python-poetry.org | python3 -
```

または、pip を使用する場合：

```shell
pip install poetry
```

## 3. 依存関係のインストール

```shell
poetry install --no-root
```

これにより、以下の依存パッケージがインストールされます：

- cycless (>=0.6.3,<0.7.0)
- networkx (>=3.6,<4.0)
- numpy (>=2.3.5,<3.0.0)
- pairlist (>=0.6,<0.7)
- matplotlib (>=3.10.7,<4.0.0)

## 4. データファイルの準備

解析に使用する`.gro`ファイルをプロジェクトディレクトリに配置してください。
デフォルトでは`00400.40.gro`が使用されます。

## 5. 実行

### cyclez

zスライスごとに、6員環のリングラベルの統計をとります。

```shell
poetry run python cyclez.py
```

実行後、以下のファイルが生成されます：

- `cycles.pdf` / `cycles.png`: サイクルサイズ別の比率グラフ
- `rings.pdf` / `rings.png`: リングコード別の比率グラフ

### cycle_dipole

6員環の実効双極子の向きをyaplotで表示します。

### grid_dipole

グリッドごとの実効双極子の向きをyaplotで表示します。
