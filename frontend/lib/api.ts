import {
  ChatRequest,
  ChatResponse,
  ChatSession,
  ChatMessage,
} from "@/types/chat";

const API_BASE_URL = "http://127.0.0.1:8000/api/v1";

export async function sendMessage(payload: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat/message`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`sendMessage failed: ${response.status} ${text}`);
  }

  return response.json();
}

export async function getChats(): Promise<ChatSession[]> {
  const response = await fetch(`${API_BASE_URL}/chat/sessions`, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`getChats failed: ${response.status} ${text}`);
  }

  return response.json();
}

export async function getMessages(chatId: string): Promise<ChatMessage[]> {
  const response = await fetch(`${API_BASE_URL}/chat/${chatId}/messages`, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`getMessages failed: ${response.status} ${text}`);
  }

  return response.json();
}