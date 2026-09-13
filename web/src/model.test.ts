import { describe, expect, it } from 'vitest';
import fixtures from './fixtures.json';
import { analyze, configFor, DEFAULTS, elementBytes, estimate, EXPERIMENTS, ms, parameterCount, roofline, stepWorkload } from './model';
import type { Precision, Settings } from './model';

describe('parity with Python infer_opt.profiling', () => {
  for (const f of fixtures) {
    it(`matches H${f.config.hidden_size}, B${f.batch}, T${f.prompt}, ${f.precision}`, () => {
      const settings = { ...DEFAULTS, hidden: f.config.hidden_size, batch: f.batch, prompt: f.prompt,
        output: f.output, precision: f.precision as Precision };
      expect(configFor(settings.hidden)).toEqual(f.config);
      expect(parameterCount(f.config)).toBe(f.parameter_count);
      const result = analyze(settings);
      expect(result.residentKv).toBe(f.resident_kv);
      expect(result.ridge).toBe(f.ridge);
      for (const step of f.steps) {
        const actual = stepWorkload(f.config, f.batch, step.new_tokens, step.context_len, elementBytes(settings.precision));
        expect(actual.flops).toEqual(step.flops);
        expect(actual.traffic).toEqual(step.traffic);
        expect(actual.intensity).toBeCloseTo(step.intensity, 10);
        expect(estimate(actual, settings).ceiling).toBeCloseTo(step.ceiling, 10);
      }
    });
  }
});

it('changes the resource limit across the ridge', () => {
  expect(roofline(50, 100, 1000)).toBe(50);
  expect(roofline(100, 100, 1000)).toBe(100);
  expect(roofline(200, 100, 1000)).toBe(100);
  const decode = analyze(DEFAULTS).decode!;
  expect(decode.bound).toBe('memory');
  expect(analyze({ ...DEFAULTS, bandwidth: 100000, compute: 0.1 }).decode!.bound).toBe('compute');
});

it('accounts for first token in prefill and grows decode context per step', () => {
  const result = analyze({ ...DEFAULTS, output: 4 });
  expect(result.decodeTimes).toHaveLength(3);
  expect(result.decodeTimes[2]).toBeGreaterThan(result.decodeTimes[0]);
  expect(result.totalMs).toBeCloseTo(result.prefill.ms + result.decodeTimes.reduce((a, b) => a + b, 0), 12);
  expect(result.tpot).toBeCloseTo(result.decodeMs / 3, 12);
  expect(result.throughput).toBeCloseTo(4 * 1000 / result.totalMs, 10);
});

it('has no decode step or TPOT when the output contains one token', () => {
  const r = analyze({ ...DEFAULTS, output: 1 });
  expect(r.decode).toBeNull();
  expect(r.tpot).toBeNull();
  expect(r.decodeTimes).toEqual([]);
  expect(r.totalMs).toBe(r.prefill.ms);
  expect(r.residentKv).toBe(2 * 16 * 1 * 256 * DEFAULTS.prompt * 2);
});

it('shares weights across the batch, while KV and FLOPs scale with batch', () => {
  const a = analyze(DEFAULTS), b = analyze({ ...DEFAULTS, batch: 32 });
  expect(b.decode!.traffic.weights).toBe(a.decode!.traffic.weights);
  expect(b.decode!.traffic.kv_read).toBe(32 * a.decode!.traffic.kv_read);
  expect(b.decode!.flops.total).toBe(32 * a.decode!.flops.total);
  expect(b.decode!.intensity).toBeGreaterThan(a.decode!.intensity);
  expect(b.throughput).toBeCloseTo(32 * DEFAULTS.output * 1000 / b.totalMs, 8);
});

it('halves bytes for FP16/BF16 without inventing higher compute capability', () => {
  const a = analyze(DEFAULTS), b = analyze({ ...DEFAULTS, precision: 'float32' });
  expect(b.decode!.traffic.total).toBe(2 * a.decode!.traffic.total);
  expect(b.decode!.computeMs).toBe(a.decode!.computeMs);
  expect(analyze({ ...DEFAULTS, precision: 'bfloat16' })).toEqual(a);
});

it('keeps the guided scenarios valid and meaningful', () => {
  for (const e of EXPERIMENTS) {
    expect(analyze(e.before).totalMs).toBeGreaterThan(0);
    expect(analyze(e.after).totalMs).toBeGreaterThan(0);
  }
  expect(analyze(EXPERIMENTS[1].after).residentKv).toBeGreaterThan(analyze(DEFAULTS).residentKv);
  expect(analyze(EXPERIMENTS[2].after).decode!.ms).toBe(analyze(DEFAULTS).decode!.ms);
});

it.each([
  { batch: 0 }, { batch: 129 }, { batch: 1.5 }, { prompt: 0 }, { prompt: 32769 },
  { output: 0 }, { output: 513 }, { hidden: 1000 }, { compute: 0 }, { bandwidth: -1 },
  { compute: NaN }, { bandwidth: Infinity }, { precision: 'int4' },
])('rejects invalid settings %o', patch => {
  expect(() => analyze({ ...DEFAULTS, ...patch } as Settings)).toThrow();
});

it('rejects invalid causal shapes', () => {
  expect(() => stepWorkload(configFor(1024), 1, 100, 50, 2)).toThrow();
  expect(() => stepWorkload(configFor(1024), 0, 1, 50, 2)).toThrow();
});

it('keeps extreme allowed settings finite', () => {
  for (const compute of [0.1, 10000]) for (const bandwidth of [1, 100000]) {
    const result = analyze({ ...DEFAULTS, batch: 128, prompt: 32768, output: 512, hidden: 4096, precision: 'float32', compute, bandwidth });
    expect(Number.isFinite(result.totalMs)).toBe(true);
    expect(result.throughput).toBeGreaterThan(0);
    expect(result.prefill.ceiling).toBeLessThanOrEqual(compute);
  }
});

it('formats tiny positive times without presenting them as zero', () => {
  expect(ms(0)).toBe('0');
  expect(ms(0.0001)).toBe('<0.001');
  expect(ms(100000000)).toBe('1.00e+8');
});
