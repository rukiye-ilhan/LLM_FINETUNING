"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Brain,
  CloudRain,
  Flame,
  Leaf,
  Pause,
  Play,
  RotateCcw,
  Sparkles,
  Trees,
  Volume2,
  Waves,
  Wind,
} from "lucide-react";

type SoundKey = "rain" | "forest" | "ocean" | "fireplace";

type SoundItem = {
  key: SoundKey;
  title: string;
  subtitle: string;
  src: string;
  icon: React.ReactNode;
};

const sounds: SoundItem[] = [
  {
    key: "rain",
    title: "Yağmur",
    subtitle: "Düşünce akışını yumuşatır",
    src: "/sounds/rain.mp3",
    icon: <CloudRain className="h-5 w-5" />,
  },
  {
    key: "forest",
    title: "Orman",
    subtitle: "Doğa hissi ve sakin odak",
    src: "/sounds/forest.mp3",
    icon: <Trees className="h-5 w-5" />,
  },
  {
    key: "ocean",
    title: "Okyanus",
    subtitle: "Nefes ritmini dengeler",
    src: "/sounds/ocean.mp3",
    icon: <Waves className="h-5 w-5" />,
  },
  {
    key: "fireplace",
    title: "Şömine",
    subtitle: "Güvenli ve sıcak atmosfer",
    src: "/sounds/fireplace.mp3",
    icon: <Flame className="h-5 w-5" />,
  },
];

const groundingSteps = [
  "Gördüğün 5 şeyi fark et.",
  "Dokunabildiğin 4 şeyi hisset.",
  "Duyduğun 3 sesi dinle.",
  "Koklayabildiğin 2 şeyi fark et.",
  "Kendine iyi gelen 1 cümle söyle.",
];

export default function WellnessPanel() {
  const [activeSound, setActiveSound] = useState<SoundKey | null>(null);
  const [breathingActive, setBreathingActive] = useState(false);
  const [groundingIndex, setGroundingIndex] = useState(0);
  const audioRefs = useRef<Record<SoundKey, HTMLAudioElement | null>>({
    rain: null,
    forest: null,
    ocean: null,
    fireplace: null,
  });

  const breathingLabel = useMemo(() => {
    if (!breathingActive) return "Başlat";
    return "Durdur";
  }, [breathingActive]);

  useEffect(() => {
    const currentAudios = audioRefs.current;

    Object.entries(currentAudios).forEach(([key, audio]) => {
      if (!audio) return;

      audio.volume = 0.45;
      audio.loop = true;

      if (key === activeSound) {
        audio.play().catch(() => {
          setActiveSound(null);
        });
      } else {
        audio.pause();
        audio.currentTime = 0;
      }
    });

    return () => {
      Object.values(currentAudios).forEach((audio) => audio?.pause());
    };
  }, [activeSound]);

  const toggleSound = (key: SoundKey) => {
    setActiveSound((prev) => (prev === key ? null : key));
  };

  return (
    <aside className="space-y-4">
      <section className="rounded-[2rem] border border-white/70 bg-white/75 p-5 shadow-[0_24px_80px_rgba(15,23,42,0.07)] backdrop-blur">
        <div className="mb-5 flex items-center justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.35em] text-sky-300">
              Rahatlama Alanı
            </p>
            <h2 className="mt-2 text-xl font-semibold text-slate-800">
              Hızlı sakinleşme araçları
            </h2>
          </div>
          <div className="rounded-2xl bg-sky-50 p-3 text-sky-500">
            <Sparkles className="h-5 w-5" />
          </div>
        </div>

        <div className="rounded-[1.8rem] bg-gradient-to-br from-sky-50 via-white to-emerald-50 p-5 text-center">
          <div className="relative mx-auto mb-4 grid h-40 w-40 place-items-center rounded-full bg-white shadow-[0_20px_70px_rgba(125,211,252,0.35)]">
            {breathingActive && <div className="pulse-ring absolute inset-0 rounded-full" />}
            <div className={`grid h-28 w-28 place-items-center rounded-full bg-sky-50 text-sky-500 shadow-inner ${breathingActive ? "breathe-orb" : ""}`}>
              <Wind className="h-8 w-8" />
            </div>
          </div>

          <h3 className="text-lg font-semibold text-slate-800">
            4-4-6 nefes egzersizi
          </h3>
          <p className="mx-auto mt-2 max-w-xs text-sm leading-6 text-slate-500">
            4 saniye nefes al, 4 saniye tut, 6 saniyede yavaşça bırak.
          </p>

          <button
            type="button"
            onClick={() => setBreathingActive((prev) => !prev)}
            className="mt-5 inline-flex items-center gap-2 rounded-2xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-slate-200 transition hover:-translate-y-0.5 hover:bg-slate-800"
          >
            {breathingActive ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {breathingLabel}
          </button>
        </div>
      </section>

      <section className="rounded-[2rem] border border-white/70 bg-white/75 p-5 shadow-[0_24px_80px_rgba(15,23,42,0.07)] backdrop-blur">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.35em] text-emerald-300">
              Doğa Sesleri
            </p>
            <h2 className="mt-2 text-lg font-semibold text-slate-800">
              Odak sesi seç
            </h2>
          </div>
          <Volume2 className="h-5 w-5 text-emerald-400" />
        </div>

        <div className="grid grid-cols-2 gap-3">
          {sounds.map((sound) => {
            const isActive = activeSound === sound.key;

            return (
              <button
                key={sound.key}
                type="button"
                onClick={() => toggleSound(sound.key)}
                className={`rounded-3xl border p-4 text-left transition hover:-translate-y-0.5 ${
                  isActive
                    ? "border-sky-200 bg-sky-50 shadow-lg shadow-sky-100"
                    : "border-slate-100 bg-white/80 hover:border-sky-100"
                }`}
              >
                <span
                  className={`mb-3 grid h-10 w-10 place-items-center rounded-2xl ${
                    isActive
                      ? "bg-sky-500 text-white"
                      : "bg-slate-50 text-slate-500"
                  }`}
                >
                  {sound.icon}
                </span>
                <span className="block text-sm font-semibold text-slate-800">
                  {sound.title}
                </span>
                <span className="mt-1 block text-xs leading-5 text-slate-500">
                  {sound.subtitle}
                </span>
              </button>
            );
          })}
        </div>

        {sounds.map((sound) => (
          <audio
            key={sound.key}
            ref={(node) => {
              audioRefs.current[sound.key] = node;
            }}
            src={sound.src}
            preload="metadata"
          />
        ))}

        <p className="mt-4 rounded-2xl bg-slate-50 px-4 py-3 text-xs leading-5 text-slate-500">
          Ses dosyalarını <b>frontend/public/sounds</b> klasörüne aynı isimlerle
          koy: rain.mp3, forest.mp3, ocean.mp3, fireplace.mp3.
        </p>
      </section>

      <section className="rounded-[2rem] border border-white/70 bg-white/75 p-5 shadow-[0_24px_80px_rgba(15,23,42,0.07)] backdrop-blur">
        <div className="flex items-start gap-3">
          <div className="rounded-2xl bg-emerald-50 p-3 text-emerald-500">
            <Brain className="h-5 w-5" />
          </div>
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.28em] text-emerald-300">
              5-4-3-2-1
            </p>
            <h2 className="mt-2 text-lg font-semibold text-slate-800">
              Grounding egzersizi
            </h2>
          </div>
        </div>

        <div className="mt-5 rounded-3xl bg-gradient-to-br from-emerald-50 to-white p-5">
          <p className="text-sm font-medium leading-7 text-slate-700">
            {groundingSteps[groundingIndex]}
          </p>
          <div className="mt-4 flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400">
              {groundingIndex + 1} / {groundingSteps.length}
            </span>
            <button
              type="button"
              onClick={() =>
                setGroundingIndex((prev) =>
                  prev === groundingSteps.length - 1 ? 0 : prev + 1
                )
              }
              className="inline-flex items-center gap-2 rounded-2xl bg-white px-4 py-2 text-xs font-semibold text-slate-700 shadow-sm"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Sonraki
            </button>
          </div>
        </div>

        <div className="mt-4 flex items-center gap-2 rounded-2xl bg-slate-50 px-4 py-3 text-xs leading-5 text-slate-500">
          <Leaf className="h-4 w-4 text-emerald-400" />
          Bu alan terapi yerine geçmez; yalnızca sakinleşme ve odak desteği sağlar.
        </div>
      </section>
    </aside>
  );
}
