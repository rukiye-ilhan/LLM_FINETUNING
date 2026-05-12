type MessageBubbleProps = {
  role: "user" | "assistant" | "error";
  content: string;
};

export default function MessageBubble({
  role,
  content,
}: MessageBubbleProps) {
  const isUser = role === "user";
  const isError = role === "error";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-5 py-4 text-sm leading-7 shadow-sm ${
          isUser
            ? "bg-slate-900 text-white rounded-br-md"
            : isError
            ? "bg-red-50 text-red-700 border border-red-200 rounded-bl-md"
            : "bg-white text-slate-800 border border-slate-200 rounded-bl-md"
        }`}
      >
        <p className="whitespace-pre-wrap">{content}</p>
      </div>
    </div>
  );
}