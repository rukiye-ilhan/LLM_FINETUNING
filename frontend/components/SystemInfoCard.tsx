import { SystemInfo } from "@/types/chat";

type SystemInfoCardProps = {
  info: SystemInfo | null;
};

export default function SystemInfoCard({ info }: SystemInfoCardProps) {
  if (!info) return null;

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      <div className="rounded-2xl border border-slate-200 bg-white p-4">
        <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Emotion
        </p>
        <p className="text-sm text-slate-800">
          {info.predicted_emotion || "-"}
        </p>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-4">
        <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Tone
        </p>
        <p className="text-sm text-slate-800">{info.tone || "-"}</p>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-4">
        <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Retrieved Topics
        </p>
        <p className="text-sm text-slate-800">
          {info.retrieved_topics?.length
            ? info.retrieved_topics.join(", ")
            : "-"}
        </p>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-4">
        <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Retrieved Docs
        </p>
        <p className="text-sm text-slate-800">
          {info.retrieved_document_count ?? 0}
        </p>
      </div>

      {info.safety_flag && (
        <div className="md:col-span-2 xl:col-span-4 rounded-2xl border border-amber-200 bg-amber-50 p-4">
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-amber-700">
            Safety Notice
          </p>
          <p className="text-sm text-amber-800">
            {info.safety_reason || "Safety-sensitive content detected."}
          </p>
        </div>
      )}
    </div>
  );
}