import {
  AuthResponse,
  AuthUser,
  ChatMessage,
  ChatRequest,
  ChatResponse,
  ChatSession,
  LoginRequest,
  RegisterRequest,
} from "@/types/chat";
import { getToken } from "@/lib/auth";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000/api/v1";

function authHeaders(): HeadersInit {
  const token = getToken();

  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${text}`);
  }

  return response.json();
}

export async function register(payload: RegisterRequest): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  return handleResponse<AuthResponse>(response);
}

export async function login(payload: LoginRequest): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  return handleResponse<AuthResponse>(response);
}

export async function getMe(): Promise<AuthUser> {
  const response = await fetch(`${API_BASE_URL}/auth/me`, {
    method: "GET",
    headers: authHeaders(),
    cache: "no-store",
  });

  return handleResponse<AuthUser>(response);
}

export async function sendMessage(payload: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat/message`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify(payload),
  });

  return handleResponse<ChatResponse>(response);
}

export async function getChats(): Promise<ChatSession[]> {
  const response = await fetch(`${API_BASE_URL}/chat/sessions`, {
    method: "GET",
    headers: authHeaders(),
    cache: "no-store",
  });

  return handleResponse<ChatSession[]>(response);
}

export async function getMessages(chatId: string): Promise<ChatMessage[]> {
  const response = await fetch(`${API_BASE_URL}/chat/${chatId}/messages`, {
    method: "GET",
    headers: authHeaders(),
    cache: "no-store",
  });

  return handleResponse<ChatMessage[]>(response);
}
