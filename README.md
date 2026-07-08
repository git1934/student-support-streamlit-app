# 児童サポート必要度ラベリング Streamlit サンプル

## 起動方法（Windows）

この版は、`numpy.core.multiarray failed to import` を避けるため、依存ライブラリのバージョンを固定しています。

### 1. フォルダに移動

```bat
cd C:\Users\DDB\Downloads\student_support_streamlit_app_fixed\student_support_streamlit_app_fixed
```

### 2. 既存ライブラリを入れ直す

```bat
python -m pip uninstall -y numpy pandas pyarrow streamlit plotly
python -m pip install --no-cache-dir --force-reinstall -r requirements.txt
```

### 3. 起動

```bat
python -m streamlit run app.py
```

`streamlit` コマンドが認識されない場合でも、`python -m streamlit run app.py` で起動できます。

## 内容

- CSVアップロードなし
- CSV出力なし
- 2026年4月の30名テストデータを内蔵
- サポート必要度ラベル：低／中／高
- 0/1フラグ表示に切り替え可能
- 変数選択、向き、重み、閾値を画面上で調整可能

## 内蔵データ

`data/student_support_dummy_summary_30students_202604.csv`


## 2026-07-07 更新

- 重み設定を小数ではなく整数に変更しました。
- 重みは 0〜5 で設定します。
- 0 はスコアに反映しない、1 は標準、2〜5 は重視として扱います。
