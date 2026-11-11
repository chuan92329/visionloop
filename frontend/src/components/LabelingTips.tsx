// "How to label faster" – opened from the annotator with the ? button or the ? key.

import { Modal } from "./ui";
import { useTour } from "./Tour";

const KEYS: [string, string][] = [
  ["Enter", "確認並跳下一張（最常用）"],
  ["1–9", "切換類別；選取物件時按，直接改它的類別"],
  ["Tab", "逐一選取每個物件，檢查模型框得對不對"],
  ["B / P / S", "畫框 / 多邊形 / 點選分割"],
  ["Del", "刪除選取的物件"],
  ["A / D", "上一張 / 下一張"],
  ["方向鍵", "微調選取的框（加 Shift 一次移 10 像素）"],
  ["Ctrl+Z", "復原"],
];

export default function LabelingTips({ onClose }: { onClose: () => void }) {
  const tour = useTour();
  const replay = () => {
    onClose();
    tour.start();
  };
  return (
    <Modal
      title="怎麼標會越來越快"
      onClose={onClose}
      footer={
        <>
          <button className="ghost" onClick={replay}>
            看一次導覽
          </button>
          <button className="primary" onClick={onClose}>
            知道了
          </button>
        </>
      }
    >
      <div className="tips">
        <section>
          <h3>一圈一圈來</h3>
          <ol className="small">
            <li>每個類別先手動標 3～5 個框，按「確認」。</li>
            <li>到「輔助標註」跑一次初步標註（Linear Probe），讓模型框剩下的圖。</li>
            <li>回來審核：只改錯的、補漏的，按 Enter 下一張。</li>
            <li>每多確認一批就再跑一次；累積約 50 張後訓練 YOLO 或 RF-DETR，之後用它標註。</li>
          </ol>
        </section>
        <section>
          <h3>省時間的習慣</h3>
          <ul className="small">
            <li><b>改，比畫快：</b>讓模型先框，人只負責修正。</li>
            <li><b>低分優先：</b>右側勾「低分優先」，先檢查模型最沒把握的框，修正這些對模型幫助最大。</li>
            <li><b>標準要一致：</b>被遮住一半要不要標、框要貼多緊，前後一致，模型才學得準。</li>
            <li><b>要輪廓就用點選分割：</b>按 S 點一下物件就有輪廓，右鍵排除不要的部分，Enter 完成。</li>
            <li><b>確認過的才會被學：</b>只有按了「確認」的圖片會用來訓練。</li>
          </ul>
        </section>
        <section>
          <h3>快捷鍵</h3>
          <div className="keys small">
            {KEYS.map(([k, v]) => (
              <div key={k} style={{ display: "contents" }}>
                <kbd>{k}</kbd>
                <span>{v}</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </Modal>
  );
}
