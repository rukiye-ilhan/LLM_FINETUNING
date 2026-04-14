import ChatBox from "@/components/ChatBox";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-gray-100 px-4 py-10">
      <div className="mx-auto flex max-w-5xl justify-center">
        <ChatBox />
      </div>
    </main>
  );
}