# 第三方軟體與授權

VisionLoop 本身以 MIT 授權釋出。以下是它使用或可以搭配使用的第三方軟體。

## 隨本專案安裝的相依套件

| 套件 | 用途 | 授權 |
| --- | --- | --- |
| FastAPI / Starlette | API 伺服器 | MIT / BSD-3-Clause |
| Uvicorn | ASGI 伺服器 | BSD-3-Clause |
| SQLAlchemy | 資料庫存取 | MIT |
| Pydantic | 資料驗證 | MIT |
| Pillow | 影像讀寫、縮圖 | MIT-CMU (HPND) |
| PyYAML | data.yaml 讀寫 | MIT |
| python-multipart | 檔案上傳 | Apache-2.0 |
| React / React DOM / React Router | 前端介面 | MIT |
| Konva / react-konva | 標註畫布 | MIT |
| pywebview（只在 Windows 由 setup_windows.ps1 安裝） | 獨立視窗 | BSD-3-Clause |
| uv（電腦上沒有 conda 時，由 setup_windows.ps1 下載到 tools\，不隨專案散布） | 安裝 Python 與建立環境 | MIT / Apache-2.0 |

以上皆允許商業使用。

### 讀取特殊影像格式（可選）

`requirements.txt` 也會安裝下面兩個套件，用來在上傳時轉換 iPhone 照片與相機 RAW 檔。程式只在上傳這些格式時才載入它們，沒有安裝也能正常使用平台。

| 套件 | 用途 | 授權 | 注意事項 |
| --- | --- | --- | --- |
| pillow-heif | 讀取 HEIC / HEIF | 套件本身 BSD-3-Clause | 官方預先編譯的安裝檔內含 x265（GPLv2）、libheif 與 libde265（LGPLv3），因此**安裝檔整體為 GPLv2**。只是自己安裝使用沒有問題；若要把它和你的程式一起打包散布（例如 Docker 映像檔），請遵守 GPLv2，或移除這個套件。 |
| rawpy | 讀取 DNG | MIT | 內含 LibRaw（LGPL-2.1 或 CDDL-1.0 擇一）。 |

介面字型使用作業系統內建的 Times New Roman 與標楷體，本專案不包含、也不散布任何字型檔；系統沒有這些字型時會改用其他襯線字型。

## 不隨本專案散布、由使用者自行安裝的模型

內建的模型外掛只會以**獨立程序**在另一個 Python 環境中執行這些模型，本專案不包含、也不 import 它們的程式碼。

| 模型 | 來源 | 授權 | 注意事項 |
| --- | --- | --- | --- |
| RF-DETR Seg（Nano / Small / Medium / Large） | [roboflow/rf-detr](https://github.com/roboflow/rf-detr)（pip：`rfdetr`） | Apache-2.0 | 分割版本。 |
| RF-DETR（Nano / Small / Medium / Large） | [roboflow/rf-detr](https://github.com/roboflow/rf-detr)（pip：`rfdetr`） | Apache-2.0 | 只使用 Apache-2.0 的尺寸；XL、2XL、Atto/Femto/Pico 為 PML-1.0，需要 Roboflow 帳號，不提供。 |
| DINOv2（Probe 外掛的骨幹） | [facebook/dinov2-with-registers-*](https://huggingface.co/facebook/dinov2-with-registers-base)，透過 transformers | Apache-2.0 | 第一次使用時從 Hugging Face 下載。 |
| SAM 2.1 | [facebook/sam2.1-hiera-*](https://huggingface.co/facebook/sam2.1-hiera-small)，透過 transformers | Apache-2.0 | 第一次使用時從 Hugging Face 下載權重。 |
| Grounding DINO | [IDEA-Research/grounding-dino-tiny / base](https://huggingface.co/IDEA-Research/grounding-dino-base)，透過 transformers | Apache-2.0 | 同上。 |
| PyTorch / torchvision / transformers | pytorch.org、Hugging Face | BSD-3-Clause / Apache-2.0 | 安裝在獨立的 `vl-models` 環境。 |
| YOLOv9 | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) | GPL-3.0 | 若你把 YOLOv9 的程式碼或修改版本一起散布，必須遵守 GPL-3.0（提供原始碼等）。 |
| RT-DETRv2 | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 | 可商用，散布時保留授權與版權聲明。 |

刻意**不使用** Ultralytics 的實作（AGPL-3.0）。

### 預訓練權重

模型權重的授權與程式碼授權是分開的，並且可能受到訓練資料集（例如 COCO 影像本身來自 Flickr，各自有不同授權）的影響。商業使用前請自行確認你使用的權重檔授權條款。

> 以上整理僅供參考，不構成法律意見。
