"use client";

import { useState } from "react";

type ChatResponse = {
  answer: string;
  predicted_emotion: string;
  tone: string;
  retrieved_topics: string[];
  retrieved_document_count: number;
};

export default function ChatBox() {
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ChatResponse | null>(null);
  const [error, setError] = useState("");

  const sendMessage = async () => {
    if (!message.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/v1/chat/message", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error("Backend response failed.");
      }

      const data: ChatResponse = await response.json();
      setResult(data);
    } catch {
      setError("Request failed. Check backend and CORS.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-4xl rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
      <h1 className="mb-2 text-3xl font-bold text-gray-900">EmpaRAG Chat</h1>
      <p className="mb-6 text-sm text-gray-600">
        RAG + Emotion + Fine-Tuned LoRA Inference Demo
      </p>

      <div className="mb-4">
        <label className="mb-2 block text-sm font-medium text-gray-700">
          User Message
        </label>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Write your message here..."
          className="min-h-[140px] w-full rounded-xl border border-gray-300 p-4 text-sm outline-none focus:border-black"
        />
      </div>

      <div className="mb-6 flex gap-3">
        <button
          onClick={sendMessage}
          disabled={loading}
          className="rounded-xl bg-black px-5 py-3 text-sm font-medium text-white disabled:opacity-50"
        >
          {loading ? "Generating..." : "Send"}
        </button>

        <button
          onClick={() => {
            setMessage("");
            setResult(null);
            setError("");
          }}
          className="rounded-xl border border-gray-300 px-5 py-3 text-sm font-medium text-gray-700"
        >
          Clear
        </button>
      </div>

      {error && (
        <div className="mb-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          <div className="rounded-2xl border border-gray-200 bg-gray-50 p-5">
            <h2 className="mb-3 text-lg font-semibold text-gray-900">Answer</h2>
            <p className="whitespace-pre-wrap text-sm leading-7 text-gray-800">
              {result.answer}
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-2xl border border-gray-200 p-5">
              <h3 className="mb-2 text-sm font-semibold text-gray-900">
                Predicted Emotion
              </h3>
              <p className="text-sm text-gray-700">{result.predicted_emotion}</p>
            </div>

            <div className="rounded-2xl border border-gray-200 p-5">
              <h3 className="mb-2 text-sm font-semibold text-gray-900">Tone</h3>
              <p className="text-sm text-gray-700">{result.tone}</p>
            </div>

            <div className="rounded-2xl border border-gray-200 p-5">
              <h3 className="mb-2 text-sm font-semibold text-gray-900">
                Retrieved Topics
              </h3>
              <p className="text-sm text-gray-700">
                {result.retrieved_topics.join(", ")}
              </p>
            </div>

            <div className="rounded-2xl border border-gray-200 p-5">
              <h3 className="mb-2 text-sm font-semibold text-gray-900">
                Retrieved Document Count
              </h3>
              <p className="text-sm text-gray-700">
                {result.retrieved_document_count}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}