"use client";

import { useEffect, useRef, useState } from "react";
import ChatSidebar from "@/components/ChatSidebar";
import MessageBubble from "@/components/MessageBubble";
import SystemInfoCard from "@/components/SystemInfoCard";
import { getChats, getMessages, sendMessage } from "@/lib/api";
import {
  ChatMessage,
  ChatSession,
  SystemInfo,
} from "@/types/chat";

export default function ChatBox() {
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [currentChatTitle, setCurrentChatTitle] = useState<string>("New Chat");

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Welcome. Share what is on your mind, and I will respond with context-aware and emotion-aware support.",
    },
  ]);

  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [error, setError] = useState("");

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const loadSessions = async () => {
    try {
      const data = await getChats();
      setSessions(data);
    } catch {
      // sessions endpoint henüz yoksa sessiz fallback
      setSessions([]);
    }
  };

  const loadChatMessages = async (chatId: string) => {
    try {
      const data = await getMessages(chatId);
      setMessages(
        data.length
          ? data
          : [
              {
                role: "assistant",
                content: "This conversation is empty.",
              },
            ]
      );
    } catch {
      setError("Could not load chat history.");
    }
  };

  useEffect(() => {
    loadSessions();
  }, []);

  const handleNewChat = () => {
    setCurrentChatId(null);
    setCurrentChatTitle("New Chat");
    setMessages([
      {
        role: "assistant",
        content:
          "Welcome. Start a new conversation whenever you are ready.",
      },
    ]);
    setSystemInfo(null);
    setError("");
    setInputText("");
  };

  const handleSelectChat = async (chatId: string) => {
    setCurrentChatId(chatId);
    const selected = sessions.find((x) => x.chat_id === chatId);
    setCurrentChatTitle(selected?.title || "Chat");
    setSystemInfo(null);
    setError("");
    await loadChatMessages(chatId);
  };

  const handleSend = async () => {
    if (!inputText.trim() || isLoading) return;

    const currentInput = inputText.trim();

    const optimisticUserMessage: ChatMessage = {
      role: "user",
      content: currentInput,
    };

    setMessages((prev) => [...prev, optimisticUserMessage]);
    setInputText("");
    setIsLoading(true);
    setError("");

    try {
      const data = await sendMessage({
        message: currentInput,
        chat_id: currentChatId,
      });

      setCurrentChatId(data.chat_id);
      setCurrentChatTitle(data.chat_title);

      setSystemInfo({
        predicted_emotion: data.predicted_emotion,
        tone: data.tone,
        retrieved_topics: data.retrieved_topics,
        retrieved_document_count: data.retrieved_document_count,
        safety_flag: data.safety_flag,
        safety_reason: data.safety_reason,
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer || "I hear you. Thank you for sharing.",
          emotion: data.predicted_emotion,
          tone: data.tone,
        },
      ]);

      await loadSessions();
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          role: "error",
          content:
            "Message could not be sent. Please check backend connection and API format.",
        },
      ]);
      setError("Request failed. Check backend and CORS.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-7xl gap-6 px-4 py-8">
      <div className="hidden lg:block">
        <ChatSidebar
          sessions={sessions}
          currentChatId={currentChatId}
          onNewChat={handleNewChat}
          onSelectChat={handleSelectChat}
        />
      </div>

      <div className="flex-1 space-y-5">
        <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <h1 className="text-3xl font-bold text-slate-900">EmpaRAG Chat</h1>
          <p className="mt-2 text-sm text-slate-600">
            Emotion-aware RAG + LoRA-powered multi-turn assistant
          </p>

          <div className="mt-4 rounded-2xl bg-slate-50 px-4 py-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Current Chat
            </p>
            <p className="mt-1 text-sm font-medium text-slate-800">
              {currentChatTitle}
            </p>
          </div>
        </div>

        <SystemInfoCard info={systemInfo} />

        <div className="rounded-3xl border border-slate-200 bg-white shadow-sm">
          <div className="max-h-[520px] space-y-4 overflow-y-auto p-6">
            {messages.map((msg, index) => (
              <MessageBubble
                key={`${msg.role}-${index}-${msg.content.slice(0, 20)}`}
                role={msg.role}
                content={msg.content}
              />
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="rounded-2xl border border-slate-200 bg-slate-50 px-5 py-4 text-sm text-slate-500">
                  EmpaRAG is thinking...
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div className="border-t border-slate-200 p-6">
            <div className="mb-3">
              <label className="mb-2 block text-sm font-medium text-slate-700">
                Your Message
              </label>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Write your message here..."
                className="min-h-[120px] w-full rounded-2xl border border-slate-300 p-4 text-sm outline-none focus:border-slate-900"
                disabled={isLoading}
              />
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                onClick={handleSend}
                disabled={isLoading}
                className="rounded-xl bg-slate-900 px-5 py-3 text-sm font-medium text-white disabled:opacity-50"
              >
                {isLoading ? "Generating..." : "Send"}
              </button>

              <button
                onClick={handleNewChat}
                className="rounded-xl border border-slate-300 px-5 py-3 text-sm font-medium text-slate-700"
              >
                New Chat
              </button>

              <button
                onClick={() => {
                  setInputText("");
                  setError("");
                }}
                className="rounded-xl border border-slate-300 px-5 py-3 text-sm font-medium text-slate-700"
              >
                Clear Input
              </button>
            </div>

            {error && (
              <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                {error}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}