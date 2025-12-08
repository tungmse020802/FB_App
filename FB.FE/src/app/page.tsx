"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type Health = "checking" | "ok" | "fail";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

export default function Home() {
  const [health, setHealth] = useState<Health>("checking");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<"beauty" | "acne">("beauty");

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        setHealth(res.ok ? "ok" : "fail");
      } catch (err) {
        console.error(err);
        setHealth("fail");
      }
    };
    check();
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (resultUrl) URL.revokeObjectURL(resultUrl);
    };
  }, [previewUrl, resultUrl]);

  const statusLabel = useMemo(() => {
    if (health === "checking") return "Checking API...";
    if (health === "ok") return "API online";
    return "API unavailable";
  }, [health]);

  const statusColor = useMemo(() => {
    if (health === "checking") return "bg-amber-400";
    if (health === "ok") return "bg-emerald-500";
    return "bg-rose-500";
  }, [health]);

  const onFileChange = (f: File | null) => {
    setError(null);
    setResultUrl(null);
    setFile(f);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(f ? URL.createObjectURL(f) : null);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Hãy chọn một ảnh trước");
      return;
    }
    setLoading(true);
    setError(null);
    setResultUrl(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const url = `${API_BASE}/beautify?mode=${mode}`;
      const res = await fetch(url, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Beautify failed");
      }
      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      setResultUrl(objectUrl);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Có lỗi xảy ra";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-950 via-zinc-900 to-black text-zinc-50">
      <div className="mx-auto flex max-w-5xl flex-col gap-10 px-6 py-12">
        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.18em] text-zinc-400">Ultimate Beauty API</p>
            <h1 className="text-3xl font-semibold text-white sm:text-4xl">Face Beautify Playground</h1>
            <p className="mt-2 text-sm text-zinc-400">
              Gửi ảnh đến FastAPI backend tại <code className="font-mono">{API_BASE}</code> và nhận lại ảnh làm đẹp.
            </p>
          </div>
          <div className="flex items-center gap-2 text-sm text-zinc-200">
            <span className={`h-2.5 w-2.5 rounded-full ${statusColor}`} />
            {statusLabel}
          </div>
        </header>

        <main className="grid gap-8 lg:grid-cols-2">
          <form
            onSubmit={handleSubmit}
            className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-6 shadow-lg shadow-black/30"
          >
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold text-white">1) Chọn ảnh</h2>
                <p className="text-sm text-zinc-400">JPEG/PNG, chỉ 1 khuôn mặt để đạt kết quả tốt nhất.</p>
              </div>
              <label className="cursor-pointer rounded-full border border-zinc-700 px-4 py-2 text-sm font-medium text-white transition hover:border-zinc-500 hover:bg-zinc-800">
                Chọn ảnh
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
                />
              </label>
            </div>

            {previewUrl ? (
              <div className="mt-4 overflow-hidden rounded-xl border border-zinc-800 bg-black/30">
                <img src={previewUrl} alt="Preview" className="w-full object-cover" />
              </div>
            ) : (
              <div className="mt-4 rounded-xl border border-dashed border-zinc-700 bg-black/20 p-6 text-center text-sm text-zinc-500">
                Chưa có ảnh nào. Kéo & thả hoặc bấm "Chọn ảnh" để bắt đầu.
              </div>
            )}

            <div className="mt-4 flex gap-3">
              <button
                type="button"
                onClick={() => setMode("beauty")}
                className={`flex-1 rounded-lg border px-4 py-2 text-sm font-medium transition ${
                  mode === "beauty"
                    ? "border-emerald-500 bg-emerald-500/20 text-emerald-200"
                    : "border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:bg-zinc-800"
                }`}
              >
                🌸 Beauty Mode
              </button>
              <button
                type="button"
                onClick={() => setMode("acne")}
                className={`flex-1 rounded-lg border px-4 py-2 text-sm font-medium transition ${
                  mode === "acne"
                    ? "border-blue-500 bg-blue-500/20 text-blue-200"
                    : "border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:bg-zinc-800"
                }`}
              >
                🔬 Acne Healing
              </button>
            </div>

            {error && <p className="mt-3 rounded-lg border border-rose-900 bg-rose-950/60 px-3 py-2 text-sm text-rose-200">{error}</p>}

            <button
              type="submit"
              disabled={loading || health !== "ok"}
              className="mt-4 w-full inline-flex h-11 items-center justify-center rounded-full bg-emerald-500 px-5 text-sm font-semibold text-emerald-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-emerald-900 disabled:text-emerald-200"
            >
              {loading ? "Đang xử lý..." : "Gửi tới API"}
            </button>
          </form>

          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-6 shadow-lg shadow-black/30">
            <h2 className="text-xl font-semibold text-white">2) Kết quả</h2>
            <p className="text-sm text-zinc-400">
              {mode === "beauty" ? "Beauty Mode: Làm mịn da + điều chỉnh màu sắc" : "Acne Mode: K-Means phát hiện mụn + frequency separation"}
            </p>
            <div className="mt-4 min-h-[280px] overflow-hidden rounded-xl border border-zinc-800 bg-black/20">
              {resultUrl ? (
                <img src={resultUrl} alt="Result" className="w-full object-cover" />
              ) : (
                <div className="flex h-full min-h-[280px] items-center justify-center text-sm text-zinc-500">
                  Chưa có ảnh kết quả.
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
