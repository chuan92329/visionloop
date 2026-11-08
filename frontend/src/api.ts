// Typed client for the VisionLoop API.

export type Status = "unlabeled" | "auto" | "in_progress" | "reviewed";
export type Split = "train" | "valid" | "test";

export interface LabelClass {
  id: number;
  name: string;
  color: string;
  order_index: number;
}

export interface ProjectStats {
  total: number;
  unlabeled: number;
  auto: number;
  in_progress: number;
  reviewed: number;
  train: number;
  valid: number;
  test: number;
  boxes: number;
}

export interface Project {
  id: number;
  name: string;
  description: string;
  created_at: string;
  split_config: { train: number; valid: number; test: number; seed: number };
  test_locked: boolean;
  test_locked_at: string | null;
  classes: LabelClass[];
  stats: ProjectStats;
  cover_image_id: number | null;
}

export interface ImageInfo {
  id: number;
  project_id: number;
  filename: string;
  width: number;
  height: number;
  status: Status;
  split: Split | null;
  group_key: string | null;
  version: number;
  box_count: number;
  min_score: number | null;
  created_at: string;
  updated_at: string;
  reviewed_at: string | null;
}

export interface Box {
  id?: number;
  class_id: number;
  x: number;
  y: number;
  w: number;
  h: number;
  source: "manual" | "auto";
  score: number | null;
  /** Outline [[x, y], ...] in image pixels; x/y/w/h are then its bounding box. */
  polygon?: [number, number][] | null;
}

export interface AnnotationState {
  image: ImageInfo;
  boxes: Box[];
}

export interface Revision {
  id: number;
  version: number;
  action: "save" | "review" | "auto_label" | "revert" | "clear";
  status: Status;
  boxes: (Box & { class_name: string })[];
  note: string;
  created_at: string;
}

export interface SplitStats {
  images: Record<Split, number>;
  boxes: Record<Split, Record<string, number>>;
  polygons: number;
  boxes_without_polygon: number;
  unassigned_reviewed: number;
  not_reviewed: number;
  test_locked: boolean;
  warnings: string[];
}

export interface Variant {
  id: string;
  name: string;
  description: string;
  min_vram_gb: number;
  params_m: number | null;
  image_size: number | null;
  image_step: number;
}

export interface PluginInfo {
  id: string;
  name: string;
  description: string;
  family: string;
  task: "detect" | "segment";
  license: string;
  homepage: string;
  dataset_format: string;
  needs_weights: boolean;
  no_weights_label: string;
  trainable: boolean;
  prompt_mode: "none" | "exemplars" | "text" | "boxes";
  default_confidence: number;
  uses_reviewed_images: boolean;
  variants: Variant[];
  available: boolean;
  status_message: string;
}

export interface PluginList {
  plugins: PluginInfo[];
  load_errors: { file: string; error: string }[];
  plugin_dir: string;
}

export interface Job {
  id: number;
  project_id: number;
  kind: "autolabel" | "train";
  plugin: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  progress: number;
  message: string;
  params: Record<string, unknown>;
  result: Record<string, unknown>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  log?: string;
}

export interface Weights {
  id: number;
  plugin: string;
  variant: string;
  name: string;
  class_names: string[];
  metrics: Record<string, unknown>;
  job_id: number | null;
  created_at: string;
}

export interface SystemInfo {
  gpus: { index: number; name: string; memory_gb: number }[];
  torch_installed: boolean;
}

export interface ImageQuery {
  status?: Status[];
  split?: Split | "none";
  q?: string;
  max_score?: number;
  sort?: "id" | "filename" | "min_score" | "updated_at" | "box_count";
  order?: "asc" | "desc";
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

function qs(params: Record<string, unknown>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    if (Array.isArray(v)) v.forEach((x) => sp.append(k, String(x)));
    else sp.append(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: {} };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    (init.headers as Record<string, string>)["Content-Type"] = "application/json";
  }
  const res = await fetch(url, init);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") msg = data.detail;
      else if (Array.isArray(data.detail))
        msg = data.detail.map((d: { msg: string }) => d.msg.replace(/^Value error, /, "")).join("；");
    } catch {
      /* not json */
    }
    throw new ApiError(res.status, msg);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  projects: () => request<Project[]>("GET", "/api/projects"),
  project: (id: number) => request<Project>("GET", `/api/projects/${id}`),
  createProject: (name: string, description: string, classes: string[]) =>
    request<Project>("POST", "/api/projects", { name, description, classes }),
  updateProject: (id: number, patch: { name?: string; description?: string }) =>
    request<Project>("PATCH", `/api/projects/${id}`, patch),
  deleteProject: (id: number) => request<void>("DELETE", `/api/projects/${id}`),

  createClass: (pid: number, name: string, color?: string) =>
    request<LabelClass>("POST", `/api/projects/${pid}/classes`, { name, color }),
  updateClass: (id: number, patch: Partial<Pick<LabelClass, "name" | "color" | "order_index">>) =>
    request<LabelClass>("PATCH", `/api/classes/${id}`, patch),
  classUsage: (id: number) => request<{ images: number; boxes: number }>("GET", `/api/classes/${id}/usage`),
  deleteClass: (id: number) => request<void>("DELETE", `/api/classes/${id}`),

  images: (pid: number, query: ImageQuery & { offset?: number; limit?: number }) =>
    request<{ items: ImageInfo[]; total: number; offset: number; limit: number }>(
      "GET",
      `/api/projects/${pid}/images${qs({ ...query })}`,
    ),
  imageIds: (pid: number, query: ImageQuery) =>
    request<number[]>("GET", `/api/projects/${pid}/image-ids${qs({ ...query })}`),
  upload: (pid: number, files: File[], groupKey?: string) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    if (groupKey) fd.append("group_key", groupKey);
    return request<{ created: ImageInfo[]; skipped: { filename: string; reason: string }[] }>(
      "POST",
      `/api/projects/${pid}/images`,
      fd,
    );
  },
  updateImage: (id: number, patch: { group_key?: string | null; split?: Split; clear_split?: boolean }) =>
    request<ImageInfo>("PATCH", `/api/images/${id}`, patch),
  setGroup: (pid: number, ids: number[], group_key: string | null) =>
    request<{ updated: number }>("POST", `/api/projects/${pid}/images/group`, { ids, group_key }),
  deleteImages: (pid: number, ids: number[]) =>
    request<{ deleted: number }>("POST", `/api/projects/${pid}/images/delete`, { ids }),
  imageUrl: (id: number) => `/api/images/${id}/file`,
  thumbUrl: (id: number) => `/api/images/${id}/thumb`,

  annotations: (imageId: number) => request<AnnotationState>("GET", `/api/images/${imageId}/annotations`),
  saveAnnotations: (imageId: number, boxes: Box[], baseVersion: number, markReviewed: boolean) =>
    request<AnnotationState>("PUT", `/api/images/${imageId}/annotations`, {
      boxes: boxes.map(({ class_id, x, y, w, h, source, score, polygon }) => ({ class_id, x, y, w, h, source, score, polygon })),
      base_version: baseVersion,
      mark_reviewed: markReviewed,
    }),
  revisions: (imageId: number) => request<Revision[]>("GET", `/api/images/${imageId}/revisions`),
  restoreRevision: (imageId: number, version: number) =>
    request<AnnotationState>("POST", `/api/images/${imageId}/revisions/${version}/restore`),

  splitStats: (pid: number) => request<SplitStats>("GET", `/api/projects/${pid}/split`),
  runSplit: (
    pid: number,
    body: {
      config?: { train: number; valid: number; test: number; seed: number };
      mode: "incremental" | "full";
      stratify: boolean;
      lock_test?: boolean;
    },
  ) => request<SplitStats>("POST", `/api/projects/${pid}/split`, body),
  lockTest: (pid: number, locked: boolean) =>
    request<SplitStats>("POST", `/api/projects/${pid}/split/lock`, { locked }),
  autoGroup: (pid: number, pattern?: string, overwrite = false) =>
    request<{ updated: number }>("POST", `/api/projects/${pid}/auto-group`, {
      ...(pattern ? { pattern } : {}),
      overwrite,
    }),
  exportUrl: (pid: number, format: "yolo" | "yolo-seg" | "coco") => `/api/projects/${pid}/export?format=${format}`,

  plugins: () => request<PluginList>("GET", "/api/plugins"),
  reloadPlugins: () => request<PluginList>("POST", "/api/plugins/reload"),
  system: () => request<SystemInfo>("GET", "/api/system"),
  jobs: (pid: number) => request<Job[]>("GET", `/api/projects/${pid}/jobs`),
  job: (id: number) => request<Job>("GET", `/api/jobs/${id}`),
  cancelJob: (id: number) => request<Job>("POST", `/api/jobs/${id}/cancel`),
  samInfo: () => request<{ model: string; cached: boolean; size_mb: number | null }>("GET", "/api/sam/info"),
  samPrepare: (imageId: number) => request<{ ready: boolean }>("POST", `/api/images/${imageId}/sam/prepare`),
  samSegment: (
    imageId: number,
    prompt: { points: [number, number, 0 | 1][]; box: [number, number, number, number] | null; exclude: [number, number][][] },
  ) =>
    request<{ polygon: [number, number][] | null; score: number | null; removed: number }>("POST", `/api/images/${imageId}/sam`, prompt),
  deleteJob: (id: number) => request<void>("DELETE", `/api/jobs/${id}`),
  clearJobs: (pid: number, statuses: Job["status"][]) =>
    request<{ deleted: number }>("POST", `/api/projects/${pid}/clear-jobs`, { statuses }),
  autolabel: (
    pid: number,
    body: {
      plugin: string;
      weights_id?: number | null;
      variant?: string;
      confidence: number;
      statuses: Status[];
      image_ids?: number[];
      polygon_detail?: "normal" | "fine";
      overwrite_polygons?: boolean;
      device: string;
      prompts?: Record<number, string>;
      max_exemplars_per_class?: number;
      probe_head?: "linear" | "mlp";
    },
  ) => request<Job>("POST", `/api/projects/${pid}/autolabel`, body),
  exemplars: (pid: number) => request<Record<string, number>>("GET", `/api/projects/${pid}/exemplars`),
  clearAuto: (pid: number, ids?: number[]) =>
    request<{ cleared: number }>("POST", `/api/projects/${pid}/clear-auto`, { ids: ids ?? null }),
  train: (
    pid: number,
    body: {
      plugin: string;
      variant: string;
      epochs: number;
      batch_size: number;
      image_size: number;
      device: string;
      init_weights_id?: number | null;
    },
  ) => request<Job>("POST", `/api/projects/${pid}/train`, body),
  weights: (pid: number) => request<Weights[]>("GET", `/api/projects/${pid}/weights`),
  importWeights: (pid: number, fd: FormData) => request<Weights>("POST", `/api/projects/${pid}/weights/import`, fd),
  deleteWeights: (id: number) => request<void>("DELETE", `/api/weights/${id}`),
};

export const STATUS_LABEL: Record<Status, string> = {
  unlabeled: "未標註",
  auto: "自動標註",
  in_progress: "編輯中",
  reviewed: "已確認",
};

export const SPLIT_LABEL: Record<Split, string> = { train: "train", valid: "valid", test: "test" };
