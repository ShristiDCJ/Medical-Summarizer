"use client";

import { useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function HomePage() {
  const [text, setText] = useState("");
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSummary("");
    setError(null);

    if (!text.trim()) {
      setError("Please enter some text to summarize.");
      return;
    }

    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        throw new Error("Failed to generate summary.");
      }

      const data = (await response.json()) as { summary: string };
      setSummary(data.summary);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main
      style={{
        width: "min(960px, 100%)",
        background: "#1e293b",
        borderRadius: "24px",
        padding: "2.5rem",
        boxShadow: "0 20px 45px rgba(2, 6, 23, 0.4)"
      }}
    >
      <h1 style={{ fontSize: "2.5rem", margin: "0 0 0.5rem" }}>Medical Transcript Summarizer</h1>
      <p style={{ margin: "0 0 2rem", color: "#cbd5f5" }}>
        Paste any medical transcript to generate a concise clinical summary using a custom transformer.
      </p>

      <form onSubmit={handleSubmit} style={{ display: "grid", gap: "1.5rem" }}>
        <label style={{ display: "grid", gap: "0.75rem" }}>
          <span style={{ fontWeight: 600, color: "#e2e8f0" }}>Transcript</span>
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            rows={10}
            style={{
              width: "100%",
              resize: "vertical",
              padding: "1rem",
              borderRadius: "16px",
              border: "1px solid rgba(148, 163, 184, 0.4)",
              fontSize: "1rem",
              background: "#0f172a",
              color: "#f1f5f9"
            }}
            placeholder="Paste or type the full transcript..."
          />
        </label>

        <button
          type="submit"
          disabled={loading}
          style={{
            justifySelf: "flex-start",
            padding: "0.75rem 1.75rem",
            borderRadius: "999px",
            border: "none",
            fontWeight: 600,
            fontSize: "1rem",
            background: loading ? "#475569" : "#38bdf8",
            color: "#0f172a",
            cursor: loading ? "not-allowed" : "pointer",
            transition: "transform 150ms ease, background 150ms ease"
          }}
        >
          {loading ? "Summarizing..." : "Generate Summary"}
        </button>
      </form>

      {error && (
        <p style={{ marginTop: "1.5rem", color: "#f87171", fontWeight: 600 }}>
          {error}
        </p>
      )}

      {summary && (
        <section
          style={{
            marginTop: "2rem",
            background: "#0f172a",
            padding: "1.5rem",
            borderRadius: "20px",
            border: "1px solid rgba(148, 163, 184, 0.3)",
            color: "#e2e8f0",
            lineHeight: 1.6
          }}
        >
          <h2 style={{ margin: "0 0 0.75rem", fontSize: "1.5rem" }}>Summary</h2>
          <p style={{ margin: 0, whiteSpace: "pre-wrap" }}>{summary}</p>
        </section>
      )}
    </main>
  );
}

