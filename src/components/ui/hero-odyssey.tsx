"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Activity, ArrowUpRight, Radio, Sparkles } from "lucide-react";

import { cn } from "@/lib/utils";

export type OdysseyTone = "positive" | "negative" | "warning" | "neutral";

export interface OdysseyMetric {
  label: string;
  value: string;
  detail: string;
  tone?: OdysseyTone;
}

interface OdysseyNode {
  label: string;
  value: string;
}

interface HeroOdysseyProps {
  eyebrow: string;
  title: string;
  description: string;
  sectionLabel: string;
  live: boolean;
  liveLabel: string;
  latestClosed: string;
  forecast: string;
  forecastDate?: string;
  forecastConfidence?: number | null;
  metrics: OdysseyMetric[];
  nodes: OdysseyNode[];
  actions: ReactNode;
}

const nodePositions = [
  "left-[45%] top-[25%]",
  "right-[7%] top-[24%]",
  "left-[49%] top-[53%]",
  "right-[4%] top-[55%]",
] as const;

const toneClasses: Record<OdysseyTone, string> = {
  positive: "text-emerald-400",
  negative: "text-red-400",
  warning: "text-amber-300",
  neutral: "text-foreground",
};

function LightningCanvas({ reducedMotion }: { reducedMotion: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("webgl", {
      alpha: true,
      antialias: false,
      powerPreference: "low-power",
    });
    if (!context) return;
    const gl: WebGLRenderingContext = context;

    const vertexSource = `
      attribute vec2 aPosition;
      void main() {
        gl_Position = vec4(aPosition, 0.0, 1.0);
      }
    `;
    const fragmentSource = `
      precision mediump float;
      uniform vec2 iResolution;
      uniform float iTime;

      float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
      }

      float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        return mix(
          mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
          mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x),
          f.y
        );
      }

      float fbm(vec2 p) {
        float value = 0.0;
        float amplitude = 0.5;
        for (int i = 0; i < 5; i++) {
          value += amplitude * noise(p);
          p = p * 2.03 + vec2(13.7, 7.9);
          amplitude *= 0.5;
        }
        return value;
      }

      void main() {
        vec2 uv = gl_FragCoord.xy / iResolution.xy;
        vec2 p = uv * 2.0 - 1.0;
        p.x *= iResolution.x / max(iResolution.y, 1.0);

        float t = iTime * 0.24;
        float bend = (fbm(vec2(p.y * 1.55 - t, t * 0.32)) - 0.5) * 0.62;
        float detail = (fbm(vec2(p.y * 5.8 + t, 4.2)) - 0.5) * 0.12;
        float distanceToBolt = abs(p.x - bend - detail);
        float core = smoothstep(0.042, 0.0, distanceToBolt);
        float halo = smoothstep(0.62, 0.0, distanceToBolt) * 0.16;
        float energy = 0.72 + 0.28 * noise(vec2(floor(iTime * 8.0), p.y));
        float verticalFade = smoothstep(-1.1, -0.35, p.y) * smoothstep(1.15, 0.08, p.y);
        vec3 orange = vec3(1.0, 0.37, 0.035);
        vec3 warmWhite = vec3(1.0, 0.86, 0.64);
        vec3 color = (orange * halo + mix(orange, warmWhite, 0.72) * core) * energy * verticalFade;
        float alpha = clamp((core + halo) * verticalFade, 0.0, 0.92);
        gl_FragColor = vec4(color, alpha);
      }
    `;

    function compile(type: number, source: string) {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    }

    const vertexShader = compile(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = compile(gl.FRAGMENT_SHADER, fragmentSource);
    if (!vertexShader || !fragmentShader) return;

    const program = gl.createProgram();
    if (!program) return;
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      gl.deleteProgram(program);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
      return;
    }

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW,
    );
    gl.useProgram(program);
    const position = gl.getAttribLocation(program, "aPosition");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    const resolution = gl.getUniformLocation(program, "iResolution");
    const time = gl.getUniformLocation(program, "iTime");
    const startedAt = performance.now();
    let animationFrame = 0;

    const resize = () => {
      const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
      const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
      const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    };

    const render = (now: number) => {
      resize();
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform2f(resolution, canvas.width, canvas.height);
      gl.uniform1f(time, reducedMotion ? 2.5 : (now - startedAt) / 1000);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      if (!reducedMotion) animationFrame = requestAnimationFrame(render);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    render(performance.now());

    return () => {
      observer.disconnect();
      cancelAnimationFrame(animationFrame);
      if (buffer) gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
    };
  }, [reducedMotion]);

  return <canvas ref={canvasRef} aria-hidden="true" className="absolute inset-0 size-full opacity-80" />;
}

function SignalNode({ node, position, index }: { node: OdysseyNode; position: string; index: number }) {
  return (
    <motion.div
      className={cn("absolute hidden items-start gap-2 xl:flex", position)}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 + index * 0.1, duration: 0.4 }}
    >
      <span className="relative mt-1.5 size-1.5 shrink-0 bg-primary shadow-[0_0_14px_rgba(247,147,26,.9)]">
        <span className="absolute -inset-1.5 border border-primary/25" />
      </span>
      <span className="max-w-44">
        <span className="eyebrow block text-primary/80">{node.label}</span>
        <span className="mt-1 block truncate font-mono text-[10px] text-white/70">{node.value}</span>
      </span>
    </motion.div>
  );
}

export function HeroOdyssey({
  eyebrow,
  title,
  description,
  sectionLabel,
  live,
  liveLabel,
  latestClosed,
  forecast,
  forecastDate,
  forecastConfidence,
  metrics,
  nodes,
  actions,
}: HeroOdysseyProps) {
  const reduceMotion = useReducedMotion() ?? false;
  const normalizedForecast = forecast.toLowerCase();
  const forecastTone = normalizedForecast === "up"
    ? "text-emerald-300"
    : normalizedForecast === "down"
      ? "text-red-300"
      : normalizedForecast === "sideway"
        ? "text-amber-300"
        : "text-muted-foreground";

  return (
    <section className="odyssey-hero relative mb-4 overflow-hidden border border-white/10 bg-[#070707]" aria-labelledby="dashboard-title">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_72%_48%,rgba(247,147,26,.12),transparent_28rem)]" />
      <div className="absolute inset-0 opacity-50 panel-grid" />
      <LightningCanvas reducedMotion={reduceMotion} />

      <div className="relative z-10 flex min-h-[390px] flex-col">
        <div className="flex min-h-16 items-center justify-between gap-3 border-b border-white/10 bg-black/35 px-4 backdrop-blur-xl sm:px-5">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid size-9 shrink-0 rotate-45 place-items-center border border-primary/60 bg-primary/10 shadow-[0_0_24px_rgba(247,147,26,.12)]">
              <Activity className="size-4 -rotate-45 text-primary" />
            </div>
            <div className="min-w-0">
              <p className="eyebrow truncate text-primary">{eyebrow}</p>
              <p className="mt-0.5 truncate font-mono text-[10px] text-white/45">PURGED OOS / UTC DAILY</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="hidden items-center gap-2 border border-white/10 bg-black/35 px-3 py-2 font-mono text-[10px] text-white/65 lg:flex">
              <Radio className={cn("size-3", live ? "text-emerald-400" : "text-amber-300")} />
              {liveLabel}
            </span>
            <span className="hidden border border-white/10 bg-black/35 px-3 py-2 font-mono text-[10px] text-white/55 sm:block">
              CLOSED {latestClosed} UTC
            </span>
            {actions}
          </div>
        </div>

        <div className="relative grid flex-1 items-center gap-8 px-5 py-8 sm:px-8 lg:grid-cols-[minmax(0,1fr)_300px] lg:px-10">
          {nodes.slice(0, 4).map((node, index) => (
            <SignalNode key={`${node.label}-${node.value}`} node={node} position={nodePositions[index]} index={index} />
          ))}

          <motion.div
            className="relative z-10 max-w-2xl"
            initial={reduceMotion ? false : { opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, ease: "easeOut" }}
          >
            <div className="mb-5 inline-flex items-center gap-2 border border-primary/25 bg-primary/8 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] text-primary">
              <Sparkles className="size-3" />
              {sectionLabel} workspace
            </div>
            <h1 id="dashboard-title" className="max-w-xl text-4xl font-semibold leading-[0.95] tracking-[-0.055em] text-white sm:text-6xl lg:text-7xl">
              {title}
            </h1>
            <p className="mt-5 max-w-xl text-sm leading-6 text-white/52 sm:text-base">{description}</p>
          </motion.div>

          <motion.div
            className="relative z-10 border border-white/12 bg-black/55 p-4 shadow-[0_24px_80px_rgba(0,0,0,.45)] backdrop-blur-xl"
            initial={reduceMotion ? false : { opacity: 0, x: 14 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.15, duration: 0.45 }}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="eyebrow">Next calibrated call</span>
              <ArrowUpRight className="size-3.5 text-primary" />
            </div>
            <strong className={cn("mt-5 block font-mono text-4xl font-semibold uppercase tracking-[-0.04em]", forecastTone)}>{forecast}</strong>
            <div className="mt-5 h-px bg-white/10">
              <div className="h-px bg-primary shadow-[0_0_12px_rgba(247,147,26,.75)]" style={{ width: `${Math.max(0, Math.min(100, (forecastConfidence ?? 0) * 100))}%` }} />
            </div>
            <div className="mt-3 flex justify-between gap-3 font-mono text-[10px] text-white/45">
              <span>{forecastDate ?? "Awaiting refresh"}</span>
              <span>{forecastConfidence == null ? "--" : `${Math.round(forecastConfidence * 100)}% score`}</span>
            </div>
          </motion.div>
        </div>

        <div className="grid grid-cols-2 border-t border-white/10 bg-black/48 lg:grid-cols-5">
          {metrics.map((metric, index) => (
            <motion.div
              key={metric.label}
              className="min-w-0 border-b border-r border-white/8 px-4 py-4 even:border-r-0 last:col-span-2 last:border-r-0 lg:col-span-1 lg:border-b-0 lg:border-r lg:even:border-r lg:last:col-span-1 lg:last:border-r-0"
              initial={reduceMotion ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.22 + index * 0.06, duration: 0.35 }}
            >
              <p className="eyebrow">{metric.label}</p>
              <strong className={cn("mt-2 block truncate font-mono text-xl font-semibold tracking-tight", toneClasses[metric.tone ?? "neutral"])}>{metric.value}</strong>
              <p className="mt-1 truncate text-[10px] text-white/38">{metric.detail}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
