# 匯入額外的模型系列（自訂外掛）

把 `.py` 檔放進這個資料夾，再到「自動標註與訓練」頁按「重新載入外掛」（或重新啟動伺服器）就會載入。檔案中每個繼承 `ModelPlugin` 的類別都會變成一個可選的模型，並依 `task` 和 `family` 出現在訓練表單。
載入失敗的外掛不會讓伺服器當掉，錯誤訊息會顯示在「自動標註與訓練」頁的右下角。

外掛是會在這台電腦上執行的程式碼，只放你信任的檔案。所以平台不提供從網頁上傳外掛，要自己放進資料夾。

也可以用環境變數 `VISIONLOOP_PLUGIN_DIR` 指定其他資料夾。

## 最小範例

```python
# backend/plugins/my_detector.py
from pathlib import Path

from app.plugins.base import ModelPlugin, Prediction, TrainResult, Variant


class MyDetector(ModelPlugin):
    id = "my-detector"                 # API 用的唯一代號
    name = "My Detector"
    family = "My Models"               # 訓練表單的「模型系列」，同系列的外掛會排在一起
    task = "detect"                    # "detect"（框）或 "segment"（多邊形）
    license = "Apache-2.0"             # 你包裝的模型程式碼的授權
    dataset_format = "yolo"            # 或 "coco"；分割模型用 "yolo-seg"
    variants = [
        # image_size：預設訓練尺寸；image_step：尺寸必須是它的倍數（預設 32）
        Variant("s", "Small", "適合 4 GB 顯卡", min_vram_gb=4, params_m=9, image_size=640),
        Variant("l", "Large", "適合 12 GB 顯卡", min_vram_gb=12, params_m=40, image_size=640),
    ]

    def train(self, ctx):
        # ctx.dataset_dir：已匯出的資料集（YOLO 格式含 data.yaml）
        # ctx.hooks.progress(0.5, "epoch 5/10")、ctx.hooks.log("...")、ctx.hooks.check_cancel()
        weights = ctx.output_dir / "best.pt"
        ...
        return TrainResult(weights, {"mAP50": 0.71, "mAP50-95": 0.48})

    def predict(self, ctx, images: list[Path]):
        # ctx.weights_path、ctx.confidence、ctx.device、ctx.class_names
        for path in images:
            # 像素座標，左上角 + 寬高；分割模型再加上 polygon=[[x, y], ...]
            boxes = [Prediction(class_index=0, x=10, y=20, w=50, h=40, score=0.9)]
            yield path, boxes
```

## 建議

- **授權隔離**：如果你包裝的框架是 GPL / AGPL，建議像內建的 YOLOv9 外掛一樣，用 `run_process()` 以獨立程序執行原專案的腳本，而不是在外掛裡 `import` 它。
- **獨立環境**：每個框架的相依套件常常互相衝突，可以各自建立 venv，再用 `python_executable("你的環境變數")` 指定直譯器。
- **取消任務**：長時間的迴圈裡呼叫 `ctx.hooks.check_cancel()`；`run_process()` 會自動處理取消。
- **測試**：可以參考 `tests/test_official_plugins.py`，用模擬原專案輸出的假腳本測試外掛的串接邏輯。
