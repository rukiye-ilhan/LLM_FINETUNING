export type ChatRole = "user" | "assistant" | "error";

export type ChatMessage = {
  role: ChatRole;
  content: string;
  emotion?: string | null;
  tone?: string | null;
  created_at?: string;
};

export type ChatSession = {
  chat_id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type SystemInfo = {
  predicted_emotion: string;
  tone: string;
  retrieved_topics: string[];
  retrieved_document_count: number;
  safety_flag: boolean;
  safety_reason?: string | null;
};

export type ChatRequest = {
  message: string;
  chat_id?: string | null;
};

export type ChatResponse = {
  chat_id: string;
  chat_title: string;
  is_new_chat: boolean;
  answer: string;
  predicted_emotion: string;
  tone: string;
  retrieved_topics: string[];
  retrieved_document_count: number;
  safety_flag: boolean;
  safety_reason?: string | null;
};

export type AuthUser = {
  user_id: string;
  name: string;
  email: string;
};

export type AuthResponse = {
  access_token: string;
  token_type: "bearer";
  user: AuthUser;
};

export type LoginRequest = {
  email: string;
  password: string;
};

export type RegisterRequest = {
  name: string;
  email: string;
  password: string;
};
