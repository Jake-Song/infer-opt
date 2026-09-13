/** Analytic parity with infer_opt.profiling; no timings are measured here. */
export type Precision = 'float32' | 'float16' | 'bfloat16';
export interface ModelConfig {
  vocab_size: number;
  hidden_size: number;
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  intermediate_size: number;
}
export interface Settings {
  batch: number;
  prompt: number;
  output: number;
  hidden: number;
  precision: Precision;
  compute: number;
  bandwidth: number;
}
export const DEFAULTS: Settings = {
  batch: 1, prompt: 1024, output: 64, hidden: 1024,
  precision: 'float16', compute: 100, bandwidth: 1000,
};
export const HIDDENS = [256, 512, 1024, 2048, 4096];
export const elementBytes = (precision: Precision) => precision === 'float32' ? 4 : 2;
export const configFor = (hidden: number): ModelConfig => ({
  vocab_size: 32000, hidden_size: hidden, n_layers: 16,
  n_heads: 16, n_kv_heads: 4, intermediate_size: hidden * 11 / 4,
});

function validateModel(c: ModelConfig) {
  if (!Object.values(c).every(v => Number.isSafeInteger(v) && v > 0)
    || c.hidden_size % c.n_heads || c.n_heads % c.n_kv_heads
    || (c.hidden_size / c.n_heads) % 2) throw new Error('Invalid model dimensions');
}

export function validateSettings(s: Settings) {
  const integerRange = (v: number, lo: number, hi: number) => Number.isInteger(v) && v >= lo && v <= hi;
  if (!integerRange(s.batch, 1, 128) || !integerRange(s.prompt, 16, 32768)
    || !integerRange(s.output, 1, 512) || !HIDDENS.includes(s.hidden)
    || !['float32', 'float16', 'bfloat16'].includes(s.precision)
    || !Number.isFinite(s.compute) || s.compute < 0.1 || s.compute > 10000
    || !Number.isFinite(s.bandwidth) || s.bandwidth < 1 || s.bandwidth > 100000)
    throw new Error('Invalid playground settings');
}

export function parameterCount(c: ModelConfig) {
  validateModel(c);
  const h = c.hidden_size, kv = c.n_kv_heads * h / c.n_heads;
  return c.n_layers * (2 * h * h + 2 * h * kv + 3 * h * c.intermediate_size + 2 * h)
    + 2 * c.vocab_size * h + h;
}

export function kvBytes(c: ModelConfig, batch: number, context: number, bytes: number) {
  return 2 * c.n_layers * batch * (c.n_kv_heads * c.hidden_size / c.n_heads) * context * bytes;
}

export function stepWorkload(c: ModelConfig, batch: number, newTokens: number, context: number, bytes: number) {
  validateModel(c);
  if (![batch, newTokens, context, bytes].every(v => Number.isSafeInteger(v) && v > 0)
    || newTokens > context) throw new Error('Invalid step');
  const h = c.hidden_size, kv = c.n_kv_heads * h / c.n_heads;
  const tokens = batch * newTokens, start = context - newTokens;
  const pairs = (context * (context + 1) - start * (start + 1)) / 2;
  const linear = c.n_layers * tokens * 2 * h * (2 * h + 2 * kv + 3 * c.intermediate_size);
  const lm_head = 2 * tokens * h * c.vocab_size;
  const attention = 4 * c.n_layers * batch * h * pairs;
  const weights = parameterCount(c) * bytes;
  const kv_read = kvBytes(c, batch, context, bytes);
  const kv_write = kvBytes(c, batch, newTokens, bytes);
  const activations = c.n_layers * tokens * 2 * (6 * h + 3 * kv + 3 * c.intermediate_size) * bytes
    + tokens * c.vocab_size * bytes;
  const flops = { linear, attention, lm_head, total: linear + attention + lm_head };
  const traffic = { weights, kv_read, kv_write, activations, total: weights + kv_read + kv_write + activations };
  return { flops, traffic, intensity: flops.total / traffic.total };
}

export type Workload = ReturnType<typeof stepWorkload>;
export function roofline(intensity: number, compute: number, bandwidth: number) {
  return Math.min(compute, intensity * bandwidth / 1000);
}
export function estimate(w: Workload, s: Settings) {
  const computeMs = w.flops.total / (s.compute * 1e12) * 1000;
  const memoryMs = w.traffic.total / (s.bandwidth * 1e9) * 1000;
  return { ...w, computeMs, memoryMs, ms: Math.max(computeMs, memoryMs),
    ceiling: roofline(w.intensity, s.compute, s.bandwidth),
    bound: computeMs >= memoryMs ? 'compute' as const : 'memory' as const };
}
export type Phase = ReturnType<typeof estimate>;

export function analyze(s: Settings) {
  validateSettings(s);
  const config = configFor(s.hidden), bytes = elementBytes(s.precision);
  const prefill = estimate(stepWorkload(config, s.batch, s.prompt, s.prompt, bytes), s);
  // Representative decode point is the first cached step. No step exists for output=1.
  const decode = s.output > 1 ? estimate(stepWorkload(config, s.batch, 1, s.prompt + 1, bytes), s) : null;
  const decodeTimes = Array.from({ length: s.output - 1 }, (_, i) =>
    estimate(stepWorkload(config, s.batch, 1, s.prompt + i + 1, bytes), s).ms);
  const decodeMs = decodeTimes.reduce((a, b) => a + b, 0);
  const totalMs = prefill.ms + decodeMs;
  return { config, prefill, decode, decodeTimes, decodeMs, totalMs,
    tpot: decodeTimes.length ? decodeMs / decodeTimes.length : null,
    throughput: s.batch * s.output / (totalMs / 1000),
    parameters: parameterCount(config), weights: parameterCount(config) * bytes,
    residentKv: kvBytes(config, s.batch, s.prompt + s.output - 1, bytes),
    ridge: s.compute * 1000 / s.bandwidth,
  };
}
export type Analysis = ReturnType<typeof analyze>;
export const number = (n: number, digits = 1) => n.toLocaleString('en-US', { maximumFractionDigits: digits });
export function memory(n: number) {
  if (n >= 1e9) return `${number(n / 1e9, 2)} GB`;
  if (n >= 1e6) return `${number(n / 1e6, 1)} MB`;
  return `${number(n / 1e3, 1)} KB`;
}
export const ms = (n: number) => n > 0 && n < 0.001 ? '<0.001' : n >= 1e6 ? n.toExponential(2) : number(n, n < 1 ? 3 : 2);

export const EXPERIMENTS = [
  { id: 'batch', label: 'Batch를 키우면?', tag: 'WEIGHT REUSE',
    description: '같은 가중치를 더 많은 요청이 함께 읽습니다. Decode 점이 오른쪽으로 이동하는 이유를 찾아보세요.',
    before: { ...DEFAULTS }, after: { ...DEFAULTS, batch: 32 } },
  { id: 'context', label: '문맥이 길어지면?', tag: 'KV CACHE',
    description: '가중치 크기는 그대로지만 KV 읽기는 늘어납니다. Decode의 메모리 구성과 TPOT을 비교해 보세요.',
    before: { ...DEFAULTS }, after: { ...DEFAULTS, prompt: 32768 } },
  { id: 'hardware', label: '연산력 vs 대역폭', tag: 'BOTTLENECK',
    description: '연산력만 2배로 올렸습니다. 이제 대역폭도 바꿔 보세요. 어느 쪽이 현재 병목을 움직이나요?',
    before: { ...DEFAULTS }, after: { ...DEFAULTS, compute: 200 } },
] as const;
