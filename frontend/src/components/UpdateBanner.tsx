// "A newer version is available" bar at the top of every page. The platform asks GitHub at most
// once a day (see backend/app/routers/updates.py); nothing shows when offline or up to date.
// The desktop app installed from a release zip updates itself: download here, then restart, and
// the launcher swaps in the new version (keeping a backup it goes back to if anything fails).

import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { api, type UpdateInfo, type UpdateStatus } from "../api";

const SKIP_KEY = "vl-skip-version";
let laterThisVisit = false; // "稍後提醒": hidden until the app is opened again

function skipped(version: string): boolean {
  try {
    return localStorage.getItem(SKIP_KEY) === version;
  } catch {
    return false;
  }
}

export default function UpdateBanner() {
  const [info, setInfo] = useState<UpdateInfo | null>(null);
  const [status, setStatus] = useState<UpdateStatus>({ state: "idle" });
  const [hidden, setHidden] = useState(laterThisVisit);
  const [error, setError] = useState("");
  const [restarting, setRestarting] = useState(false);
  // the annotator fills the window and is where people focus: the bar waits for the other pages
  const annotating = useLocation().pathname.includes("/annotate/");

  useEffect(() => {
    api
      .updateCheck()
      .then((i) => {
        setInfo(i);
        setStatus(i.install);
      })
      .catch(() => {});
  }, []);

  // follow the download while it runs
  useEffect(() => {
    if (status.state !== "downloading") return;
    const t = window.setInterval(() => {
      api.updateStatus().then(setStatus).catch(() => {});
    }, 800);
    return () => window.clearInterval(t);
  }, [status.state]);

  if (!info || annotating) return null;

  // how the last update went, shown once
  if (info.last) {
    const close = () => {
      api.updateAck().catch(() => {});
      setInfo({ ...info, last: null });
    };
    return (
      <div className={`update-banner ${info.last.ok ? "" : "failed"}`} role="status">
        {info.last.ok ? (
          <span>
            已更新到 <b>{info.last.version}</b>
          </span>
        ) : (
          <span>
            更新到 <b>{info.last.version}</b> 沒有成功，目前仍是 v{info.current}：{info.last.message}
          </span>
        )}
        {!info.last.ok && (
          <a className="btn" href={info.url} target="_blank" rel="noreferrer">
            手動下載
          </a>
        )}
        <span className="spacer" />
        <button className="ghost" onClick={close}>
          關閉
        </button>
      </div>
    );
  }

  if (hidden || !info.newer || !info.latest || skipped(info.latest)) return null;

  const later = () => {
    laterThisVisit = true;
    setHidden(true);
  };
  const skip = () => {
    try {
      localStorage.setItem(SKIP_KEY, info.latest!);
    } catch {
      /* private window: only for this visit */
    }
    later();
  };
  const download = async () => {
    setError("");
    try {
      setStatus(await api.updateDownload());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  const restart = async () => {
    setError("");
    try {
      await api.updateRestart();
      setRestarting(true); // the window closes and VisionLoop opens again with the new version
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const percent = Math.round((status.progress ?? 0) * 100);
  const message = error || (status.state === "error" ? status.message ?? "下載失敗" : "");

  return (
    <div className="update-banner" role="status">
      <span>
        有新版本 <b>{info.latest}</b> 可以更新（目前 v{info.current}）
      </span>
      <a className={`btn ${info.can_install ? "" : "primary"}`} href={info.url} target="_blank" rel="noreferrer">
        查看更新內容
      </a>
      {info.can_install && restarting && <span>正在重新啟動並安裝更新…</span>}
      {info.can_install && !restarting && status.state === "downloading" && (
        <span className="update-progress">
          正在下載 {percent}%
          <span className="bar">
            <span style={{ width: `${percent}%` }} />
          </span>
        </span>
      )}
      {info.can_install && !restarting && status.state === "ready" && (
        <button className="primary" onClick={restart} title="VisionLoop 會關閉並重新開啟，專案與資料都不受影響">
          重新啟動並更新
        </button>
      )}
      {info.can_install && !restarting && (status.state === "idle" || status.state === "error") && (
        <button className="primary" onClick={download} title="在背景下載，下載完成後再決定何時重新啟動">
          {status.state === "error" ? "重新下載" : "立即更新"}
        </button>
      )}
      {message && <span className="update-error">{message}</span>}
      <span className="spacer" />
      {!restarting && (
        <>
          <button className="ghost" onClick={later}>
            稍後提醒
          </button>
          <button className="ghost" onClick={skip}>
            略過這個版本
          </button>
        </>
      )}
    </div>
  );
}
