import { ChatSession } from "@/types/chat";

type ChatSidebarProps = {
  sessions: ChatSession[];
  currentChatId: string | null;
  onNewChat: () => void;
  onSelectChat: (chatId: string) => void;
};

export default function ChatSidebar({
  sessions,
  currentChatId,
  onNewChat,
  onSelectChat,
}: ChatSidebarProps) {
  return (
    <aside className="w-full max-w-xs rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Chats</h2>
          <p className="text-xs text-slate-500">Conversation history</p>
        </div>

        <button
          onClick={onNewChat}
          className="rounded-xl bg-slate-900 px-3 py-2 text-xs font-medium text-white"
        >
          New Chat
        </button>
      </div>

      <div className="space-y-2">
        {sessions.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-slate-200 p-4 text-sm text-slate-500">
            No saved chats yet.
          </div>
        ) : (
          sessions.map((session) => {
            const isActive = session.chat_id === currentChatId;

            return (
              <button
                key={session.chat_id}
                onClick={() => onSelectChat(session.chat_id)}
                className={`w-full rounded-2xl border p-4 text-left transition ${
                  isActive
                    ? "border-slate-900 bg-slate-50"
                    : "border-slate-200 bg-white hover:bg-slate-50"
                }`}
              >
                <p className="line-clamp-2 text-sm font-medium text-slate-800">
                  {session.title}
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  {new Date(session.updated_at).toLocaleString()}
                </p>
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}