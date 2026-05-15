"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Loader2,
  LogOut,
  MessageCircle,
  Plus,
  Send,
  ShieldCheck,
  Sparkles,
  Trash2,
} from "lucide-react";
import ChatSidebar from "@/components/ChatSidebar";
import MessageBubble from "@/components/MessageBubble";
import SystemInfoCard from "@/components/SystemInfoCard";
import WellnessPanel from "@/components/WellnessPanel";
import { getChats, getMessages, sendMessage } from "@/lib/api";
import { AuthUser, ChatMessage, ChatSession, SystemInfo } from "@/types/chat";

type ChatBoxProps = {
  user: AuthUser;
  onLogout: () => void;
};

const welcomeMessage: ChatMessage = {
  role: "assistant",
  content:
    "Hoş geldin. Bugün senin için sakin ve berrak bir alan hazırladım. Zihnini boşaltmak istediğin her şeyi buraya bırakabilirsin.",
};

const starterPrompts = [
  "I have been feeling overwhelmed lately and I want to talk about what has been affecting me.",
  "I feel anxious and I need a calming response.",
  "Help me understand what I am feeling right now.",
];

export default function ChatBox({ user, onLogout }: ChatBoxProps) {
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [currentChatTitle, setCurrentChatTitle] = useState("Yeni Oturum");
  const [messages, setMessages] = useState<ChatMessage[]>([welcomeMessage]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [error, setError] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const loadSessions = useCallback(async () => {
    try {
      const data = await getChats();
      setSessions(data);
    } catch (err) {
      console.error(err);
      setSessions([]);
    }
  }, []);

  const resetChatState = useCallback(() => {
    setCurrentChatId(null);
    setCurrentChatTitle("Yeni Oturum");
    setMessages([welcomeMessage]);
    setSessions([]);
    setSystemInfo(null);
    setError("");
    setInputText("");
    setIsLoading(false);
  }, []);

  useEffect(() => {
    resetChatState();
    loadSessions();
  }, [user.user_id, resetChatState, loadSessions]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const loadChatMessages = async (chatId: string) => {
    try {
      const data = await getMessages(chatId);
      setMessages(
        data.length ? data : [{ role: "assistant", content: "Bu konuşma henüz boş." }]
      );
    } catch (err) {
      console.error(err);
      setError("Konuşma geçmişi yüklenemedi.");
    }
  };

  const handleNewChat = () => {
    setCurrentChatId(null);
    setCurrentChatTitle("Yeni Oturum");
    setMessages([
      {
        role: "assistant",
        content:
          "Yeni bir alan açıldı. Hazır olduğunda aklından geçenleri yazabilirsin.",
      },
    ]);
    setSystemInfo(null);
    setError("");
    setInputText("");
  };

  const handleSelectChat = async (chatId: string) => {
    setCurrentChatId(chatId);
    const selected = sessions.find((x) => x.chat_id === chatId);
    setCurrentChatTitle(selected?.title || "Konuşma");
    setSystemInfo(null);
    setError("");
    await loadChatMessages(chatId);
  };

  const handleSend = async (messageOverride?: string) => {
    const text = messageOverride ?? inputText;
    if (!text.trim() || isLoading) return;

    const currentInput = text.trim();

    setMessages((prev) => [...prev, { role: "user", content: currentInput }]);
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
        retrieval_used: data.retrieval_used,
        retrieval_strategy: data.retrieval_strategy,
        memory_turn_count: data.memory_turn_count,
        safety_flag: data.safety_flag,
        safety_reason: data.safety_reason,
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer || "Seni duyuyorum. Paylaştığın için teşekkür ederim.",
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
            "Mesaj gönderilemedi. Backend bağlantısını, token bilgisini veya CORS ayarını kontrol et.",
        },
      ]);
      setError("Request başarısız. Backend açık mı ve giriş tokenı geçerli mi kontrol et.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#f7faf8] text-slate-900">
      <div className="pointer-events-none absolute left-24 top-16 h-72 w-72 rounded-full bg-sky-100/60 blur-3xl" />
      <div className="pointer-events-none absolute bottom-24 right-36 h-80 w-80 rounded-full bg-emerald-100/70 blur-3xl" />

      <div className="relative mx-auto grid w-full max-w-[1500px] grid-cols-1 gap-6 px-4 py-6 lg:grid-cols-[260px_minmax(0,1fr)_360px] lg:px-6">
        <div className="hidden lg:block">
          <div className="sticky top-6">
            <ChatSidebar
              sessions={sessions}
              currentChatId={currentChatId}
              onNewChat={handleNewChat}
              onSelectChat={handleSelectChat}
            />
          </div>
        </div>

        <section className="min-w-0 space-y-5">
          <header className="rounded-[2rem] border border-white/70 bg-white/75 p-6 shadow-[0_24px_80px_rgba(15,23,42,0.07)] backdrop-blur">
            <div className="flex flex-col gap-5 md:flex-row md:items-start md:justify-between">
              <div>
                <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-sky-100 bg-sky-50 px-4 py-2 text-xs font-semibold uppercase tracking-[0.25em] text-sky-500">
                  <Sparkles className="h-3.5 w-3.5" />
                  EmpaRAG
                </div>

                <h1 className="text-3xl font-semibold tracking-tight text-slate-900 md:text-5xl">
                  Duygu farkındalıklı destek alanı
                </h1>

                <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-500 md:text-base">
                  Bu oturum <b>{user.name}</b> kullanıcısına özel. Chat geçmişi
                  sadece giriş yapan kullanıcının tokenı ile listelenir.
                </p>
              </div>

              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={handleNewChat}
                  className="inline-flex items-center justify-center gap-2 rounded-2xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-slate-200 transition hover:-translate-y-0.5 hover:bg-slate-800"
                >
                  <Plus className="h-4 w-4" />
                  Yeni oturum
                </button>

                <button
                  type="button"
                  onClick={onLogout}
                  className="inline-flex items-center justify-center gap-2 rounded-2xl border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
                >
                  <LogOut className="h-4 w-4" />
                  Çıkış
                </button>
              </div>
            </div>

            <div className="mt-6 grid gap-3 md:grid-cols-3">
              <div className="rounded-3xl bg-white/70 p-4">
                <p className="text-xs font-bold uppercase tracking-[0.25em] text-slate-400">
                  Aktif Oturum
                </p>
                <p className="mt-2 truncate text-sm font-semibold text-slate-800">
                  {currentChatTitle}
                </p>
              </div>

              <div className="rounded-3xl bg-white/70 p-4">
                <p className="text-xs font-bold uppercase tracking-[0.25em] text-slate-400">
                  Kullanıcı
                </p>
                <p className="mt-2 truncate text-sm font-semibold text-slate-800">
                  {user.email}
                </p>
              </div>

              <div className="rounded-3xl bg-white/70 p-4">
                <p className="text-xs font-bold uppercase tracking-[0.25em] text-slate-400">
                  Güvenlik
                </p>
                <p className="mt-2 inline-flex items-center gap-2 text-sm font-semibold text-slate-800">
                  <ShieldCheck className="h-4 w-4 text-emerald-500" />
                  Auth + private chat
                </p>
              </div>
            </div>
          </header>

          <SystemInfoCard info={systemInfo} />

          <div className="rounded-[2rem] border border-white/70 bg-white/80 shadow-[0_24px_80px_rgba(15,23,42,0.07)] backdrop-blur">
            <div className="flex items-center gap-3 border-b border-slate-100 px-5 py-4">
              <div className="grid h-11 w-11 place-items-center rounded-2xl bg-slate-900 text-white">
                <MessageCircle className="h-5 w-5" />
              </div>
              <div>
                <h2 className="font-semibold text-slate-900">Sohbet</h2>
                <p className="text-xs text-slate-500">
                  Yaz, sistem duygu + bağlam + ton üretimini göstersin.
                </p>
              </div>
            </div>

            <div className="max-h-[560px] min-h-[420px] space-y-4 overflow-y-auto p-5">
              {messages.map((msg, index) => (
                <MessageBubble
                  key={`${msg.role}-${index}-${msg.content.slice(0, 20)}`}
                  role={msg.role}
                  content={msg.content}
                />
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="inline-flex items-center gap-3 rounded-3xl border border-slate-100 bg-white px-5 py-4 text-sm text-slate-500 shadow-sm">
                    <Loader2 className="h-4 w-4 animate-spin text-sky-500" />
                    EmpaRAG bağlamı getiriyor ve tonu ayarlıyor...
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            <div className="border-t border-slate-100 p-5">
              <div className="mb-4 flex flex-wrap gap-2">
                {starterPrompts.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => setInputText(prompt)}
                    className="rounded-full border border-slate-100 bg-slate-50 px-4 py-2 text-xs font-medium text-slate-600 transition hover:border-sky-100 hover:bg-sky-50"
                  >
                    {prompt.slice(0, 58)}...
                  </button>
                ))}
              </div>

              <div className="rounded-[1.8rem] border border-slate-100 bg-slate-50/70 p-3">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder="Düşüncelerini buraya bırakabilirsin..."
                  className="min-h-[118px] w-full resize-none rounded-[1.4rem] border border-white bg-white p-4 text-sm leading-7 text-slate-700 outline-none transition focus:border-sky-200 focus:ring-4 focus:ring-sky-50"
                  disabled={isLoading}
                />

                <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                  <div className="text-xs text-slate-400">
                    Enter gönderir, Shift + Enter yeni satır açar.
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setInputText("");
                        setError("");
                      }}
                      className="inline-flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-600 transition hover:bg-slate-50"
                    >
                      <Trash2 className="h-4 w-4" />
                      Temizle
                    </button>

                    <button
                      type="button"
                      onClick={() => handleSend()}
                      disabled={isLoading || !inputText.trim()}
                      className="inline-flex items-center gap-2 rounded-2xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-slate-200 transition hover:-translate-y-0.5 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {isLoading ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin" />
                          Üretiliyor
                        </>
                      ) : (
                        <>
                          <Send className="h-4 w-4" />
                          Gönder
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>

              {error && (
                <div className="mt-4 rounded-2xl border border-red-100 bg-red-50 p-4 text-sm leading-6 text-red-700">
                  {error}
                </div>
              )}
            </div>
          </div>
        </section>

        <div className="lg:sticky lg:top-6 lg:self-start">
          <WellnessPanel />
        </div>
      </div>
    </main>
  );
}
