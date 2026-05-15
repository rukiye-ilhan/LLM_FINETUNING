"use client";

import { useState } from "react";
import { Brain, Lock, LogIn, Mail, Sparkles, UserPlus } from "lucide-react";
import { login, register } from "@/lib/api";
import { saveAuth } from "@/lib/auth";
import { AuthUser } from "@/types/chat";

type AuthPageProps = {
  onAuthSuccess: (user: AuthUser) => void;
};

export default function AuthPage({ onAuthSuccess }: AuthPageProps) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("demo@emparag.ai");
  const [password, setPassword] = useState("123456");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const isRegister = mode === "register";

  const handleSubmit = async () => {
    if (!email.trim() || !password.trim()) {
      setError("Email ve şifre zorunlu.");
      return;
    }

    if (isRegister && !name.trim()) {
      setError("Kayıt için isim zorunlu.");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      const result = isRegister
        ? await register({ name, email, password })
        : await login({ email, password });

      saveAuth(result.access_token, result.user);
      onAuthSuccess(result.user);
    } catch (err) {
      console.error(err);
      setError(
        isRegister
          ? "Kayıt başarısız. Email daha önce kullanılmış olabilir."
          : "Giriş başarısız. Email veya şifre hatalı olabilir."
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="relative grid min-h-screen place-items-center overflow-hidden bg-[#f7faf8] px-4 py-10">
      <div className="pointer-events-none absolute left-16 top-16 h-72 w-72 rounded-full bg-sky-100/70 blur-3xl" />
      <div className="pointer-events-none absolute bottom-16 right-16 h-80 w-80 rounded-full bg-emerald-100/80 blur-3xl" />

      <section className="relative grid w-full max-w-6xl overflow-hidden rounded-[2.5rem] border border-white/70 bg-white/75 shadow-[0_30px_100px_rgba(15,23,42,0.10)] backdrop-blur lg:grid-cols-[1.05fr_0.95fr]">
        <div className="p-8 md:p-12">
          <div className="mb-8 inline-flex items-center gap-2 rounded-full bg-sky-50 px-4 py-2 text-xs font-bold uppercase tracking-[0.28em] text-sky-500">
            <Sparkles className="h-4 w-4" />
            EmpaRAG
          </div>

          <h1 className="text-4xl font-semibold tracking-tight text-slate-900 md:text-6xl">
            Kişisel ve güvenli destek alanın
          </h1>

          <p className="mt-5 max-w-xl text-base leading-8 text-slate-500">
            Her kullanıcı kendi hesabıyla giriş yapar. Chat geçmişi kullanıcıya
            özel tutulur; başka kullanıcıların konuşmaları listelenmez.
          </p>

          <div className="mt-10 grid gap-4 md:grid-cols-3">
            {[
              ["Private Chat", "Kullanıcı bazlı oturum geçmişi"],
              ["Emotion-Aware", "Duygu tonuna göre yanıt"],
              ["RAG + LoRA", "Bağlam destekli üretim"],
            ].map(([title, desc]) => (
              <div key={title} className="rounded-3xl bg-white/80 p-5 shadow-sm">
                <Brain className="mb-4 h-5 w-5 text-sky-500" />
                <h3 className="font-semibold text-slate-800">{title}</h3>
                <p className="mt-2 text-sm leading-6 text-slate-500">{desc}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="border-t border-white/80 bg-white/65 p-8 md:p-12 lg:border-l lg:border-t-0">
          <div className="mb-6 flex rounded-2xl bg-slate-100 p-1">
            <button
              type="button"
              onClick={() => setMode("login")}
              className={`flex-1 rounded-xl px-4 py-3 text-sm font-semibold transition ${
                !isRegister ? "bg-white text-slate-900 shadow-sm" : "text-slate-500"
              }`}
            >
              Giriş Yap
            </button>
            <button
              type="button"
              onClick={() => setMode("register")}
              className={`flex-1 rounded-xl px-4 py-3 text-sm font-semibold transition ${
                isRegister ? "bg-white text-slate-900 shadow-sm" : "text-slate-500"
              }`}
            >
              Kayıt Ol
            </button>
          </div>

          <h2 className="text-2xl font-semibold text-slate-900">
            {isRegister ? "Yeni hesap oluştur" : "Hesabına giriş yap"}
          </h2>

          <div className="mt-7 space-y-4">
            {isRegister && (
              <label className="block">
                <span className="mb-2 block text-sm font-semibold text-slate-600">
                  İsim
                </span>
                <div className="flex items-center gap-3 rounded-2xl border border-slate-100 bg-white px-4 py-3">
                  <UserPlus className="h-5 w-5 text-slate-400" />
                  <input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Rukiye İlhan"
                    className="w-full bg-transparent text-sm outline-none"
                  />
                </div>
              </label>
            )}

            <label className="block">
              <span className="mb-2 block text-sm font-semibold text-slate-600">
                Email
              </span>
              <div className="flex items-center gap-3 rounded-2xl border border-slate-100 bg-white px-4 py-3">
                <Mail className="h-5 w-5 text-slate-400" />
                <input
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="demo@emparag.ai"
                  className="w-full bg-transparent text-sm outline-none"
                />
              </div>
            </label>

            <label className="block">
              <span className="mb-2 block text-sm font-semibold text-slate-600">
                Şifre
              </span>
              <div className="flex items-center gap-3 rounded-2xl border border-slate-100 bg-white px-4 py-3">
                <Lock className="h-5 w-5 text-slate-400" />
                <input
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  type="password"
                  placeholder="En az 6 karakter"
                  className="w-full bg-transparent text-sm outline-none"
                />
              </div>
            </label>
          </div>

          {error && (
            <div className="mt-5 rounded-2xl border border-red-100 bg-red-50 p-4 text-sm text-red-700">
              {error}
            </div>
          )}

          <button
            type="button"
            onClick={handleSubmit}
            disabled={isLoading}
            className="mt-7 inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-slate-900 px-5 py-4 text-sm font-semibold text-white shadow-lg shadow-slate-200 transition hover:-translate-y-0.5 hover:bg-slate-800 disabled:opacity-50"
          >
            <LogIn className="h-4 w-4" />
            {isLoading ? "İşleniyor..." : isRegister ? "Kayıt Ol" : "Giriş Yap"}
          </button>
        </div>
      </section>
    </main>
  );
}
