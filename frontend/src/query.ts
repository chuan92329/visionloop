import type { ImageQuery, Split, Status } from "./api";

const STATUSES: Status[] = ["unlabeled", "auto", "in_progress", "reviewed"];
const SPLITS = ["train", "valid", "test", "none"];
const SORTS = ["id", "filename", "min_score", "updated_at", "box_count"];

/** The image filter lives in the URL so the annotator can walk the same queue. */
export function queryFromSearch(sp: URLSearchParams): ImageQuery {
  const status = sp.getAll("status").filter((s): s is Status => STATUSES.includes(s as Status));
  const split = sp.get("split");
  const sort = sp.get("sort");
  const q: ImageQuery = {};
  if (status.length) q.status = status;
  if (split && SPLITS.includes(split)) q.split = split as Split | "none";
  if (sort && SORTS.includes(sort)) q.sort = sort as ImageQuery["sort"];
  if (sp.get("order") === "desc") q.order = "desc";
  if (sp.get("q")) q.q = sp.get("q")!;
  return q;
}

export function searchFromQuery(q: ImageQuery): string {
  const sp = new URLSearchParams();
  q.status?.forEach((s) => sp.append("status", s));
  if (q.split) sp.set("split", q.split);
  if (q.sort && q.sort !== "id") sp.set("sort", q.sort);
  if (q.order === "desc") sp.set("order", "desc");
  if (q.q) sp.set("q", q.q);
  const s = sp.toString();
  return s ? `?${s}` : "";
}
