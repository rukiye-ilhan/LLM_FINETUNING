"use client";

import { useEffect, useState } from "react";
import AuthPage from "@/components/AuthPage";
import ChatBox from "@/components/ChatBox";
import { clearAuth, getStoredUser, getToken } from "@/lib/auth";
import { getMe } from "@/lib/api";
import { AuthUser } from "@/types/chat";

export default function AuthGate() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    const bootstrap = async () => {
      const token = getToken();
      const storedUser = getStoredUser();

      if (!token || !storedUser) {
        setIsChecking(false);
        return;
      }

      try {
        const freshUser = await getMe();
        setUser(freshUser);
      } catch {
        clearAuth();
        setUser(null);
      } finally {
        setIsChecking(false);
      }
    };

    bootstrap();
  }, []);

  if (isChecking) {
    return (
      <main className="grid min-h-screen place-items-center bg-[#f7faf8] text-slate-500">
        EmpaRAG hazırlanıyor...
      </main>
    );
  }

  if (!user) {
    return <AuthPage onAuthSuccess={setUser} />;
  }

  return (
    <ChatBox
      key={user.user_id}
      user={user}
      onLogout={() => {
        clearAuth();
        setUser(null);
      }}
    />
  );
}
