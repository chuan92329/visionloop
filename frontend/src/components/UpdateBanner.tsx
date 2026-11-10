// "A newer version is available" bar at the top of every page. The platform asks GitHub at most
// once a day (see backend/app/routers/updates.py); nothing shows when offline or up to date.

import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { api, type UpdateInfo } from "../api";

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
  const [hidden, setHidden] = useState(laterThisVisit);
  // the annotator fills the window and is where people focus: the bar waits for the other pages
  const annotating = useLocation().pathname.includes("/annotate/");

  useEffect(() => {
    if (laterThisVisit) return;
    api.updateCheck().then(setInfo).catch(() => {});
  }, []);

  if (hidden || annotating || !info?.newer || !info.latest || skipped(info.latest)) return null;

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

  return (
    <div className="update-banner" role="status">
      <span>
        有新版本 <b>{info.latest}</b> 可以更新（目前 v{info.current}）
      </span>
      <a className="btn primary" href={info.url} target="_blank" rel="noreferrer">
        查看更新內容
      </a>
      <span className="spacer" />
      <button className="ghost" onClick={later}>
        稍後提醒
      </button>
      <button className="ghost" onClick={skip}>
        略過這個版本
      </button>
    </div>
  );
}
