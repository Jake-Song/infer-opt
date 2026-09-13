import { useEffect, useState } from 'react';
import type { Analysis, Phase, Settings } from './model';
import { memory, ms, number, roofline } from './model';

export function Roofline({ result, settings, baseline, baselineSettings }: {
  result: Analysis; settings: Settings; baseline: Analysis | null; baselineSettings: Settings | null;
}) {
  const [selected, setSelected] = useState<'prefill' | 'decode'>('decode');
  const phase = selected === 'decode' ? result.decode ?? result.prefill : result.prefill;
  const phaseName = selected === 'decode' && result.decode ? 'Decode' : 'Prefill';
  const allPhases = [result.prefill, result.decode, baseline?.prefill, baseline?.decode].filter(p => p != null);
  const minX = Math.min(0.1, ...allPhases.map(p => p.intensity / 2), result.ridge / 10, (baseline?.ridge ?? result.ridge) / 10);
  const maxX = Math.max(1000, result.ridge * 4, (baseline?.ridge ?? 1) * 4, ...allPhases.map(p => p.intensity * 3));
  const xLo = Math.floor(Math.log10(minX)), xHi = Math.ceil(Math.log10(maxX));
  const maxY = Math.max(settings.compute, baselineSettings?.compute ?? 0) * 2;
  const minY = Math.min(roofline(10 ** xLo, settings.compute, settings.bandwidth),
    baselineSettings ? roofline(10 ** xLo, baselineSettings.compute, baselineSettings.bandwidth) : Infinity,
    ...allPhases.map(p => p.ceiling)) / 2;
  const yLo = Math.floor(Math.log10(minY)), yHi = Math.ceil(Math.log10(maxY));
  const x = (v: number) => 72 + (Math.log10(v) - xLo) / (xHi - xLo) * 642;
  const y = (v: number) => 285 - (Math.log10(v) - yLo) / (yHi - yLo) * 252;
  const ticksX = Array.from({ length: xHi - xLo + 1 }, (_, i) => 10 ** (i + xLo));
  const ticksY = Array.from({ length: yHi - yLo + 1 }, (_, i) => 10 ** (i + yLo));
  const tick = (v: number) => v >= 10000 || v < 0.001 ? `10^${Math.round(Math.log10(v))}` : number(v, 4);
  const line = (s: Settings) => {
    const ridge = s.compute * 1000 / s.bandwidth;
    const xx = [10 ** xLo, Math.max(10 ** xLo, Math.min(10 ** xHi, ridge)), 10 ** xHi];
    return xx.map((v, i) => `${i ? 'L' : 'M'}${x(v)},${y(roofline(v, s.compute, s.bandwidth))}`).join(' ');
  };
  return <>
    <div className="chart-legend"><span><i className="dot cyan" />Prefill</span><span><i className="dot amber" />Decode</span><span><i className="line-swatch" />이론적 상한</span>{baseline && <span><i className="line-swatch dashed" />저장된 기준</span>}</div>
    <svg className="roofline" viewBox="0 0 760 330" role="img" aria-label="Roofline 그래프: 가로축 arithmetic intensity, 세로축 이론적 TFLOP/s 상한">
      <defs><linearGradient id="roof-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#70d2b4" stopOpacity=".09" /><stop offset="100%" stopColor="#70d2b4" stopOpacity="0" /></linearGradient></defs>
      {ticksY.map(v => <g key={v}><line x1="72" x2="714" y1={y(v)} y2={y(v)} className="grid-line" /><text x="59" y={y(v) + 4} textAnchor="end" className="axis-tick">{tick(v)}</text></g>)}
      {ticksX.map(v => <g key={v}><line x1={x(v)} x2={x(v)} y1="33" y2="285" className="grid-line vertical" /><text x={x(v)} y="306" textAnchor="middle" className="axis-tick">{tick(v)}</text></g>)}
      <path d={`${line(settings)} L714,285 L72,285 Z`} fill="url(#roof-fill)" />
      <line x1={x(result.ridge)} x2={x(result.ridge)} y1="33" y2="285" className="ridge-line" />
      <text x={Math.max(140, Math.min(645, x(result.ridge)))} y="22" textAnchor="middle" className="ridge-label">RIDGE · {number(result.ridge)} FLOP/B</text>
      {baselineSettings && <path d={line(baselineSettings)} className="baseline-roof" />}
      <path d={line(settings)} className="ceiling-line" />
      {baseline && [baseline.prefill, baseline.decode].map((p, i) => p && <circle key={i} cx={x(p.intensity)} cy={y(p.ceiling)} r="7" className="baseline-point"><title>기준 {i ? 'Decode' : 'Prefill'}: {number(p.intensity)} FLOP/B · {number(p.ceiling)} TFLOP/s</title></circle>)}
      {(['prefill', 'decode'] as const).map(name => {
        const p = result[name];
        if (!p) return null;
        const active = phaseName.toLowerCase() === name;
        return <g key={name} className={`chart-point ${name}`} role="button" tabIndex={0}
          aria-label={`${name === 'prefill' ? 'Prefill' : 'Decode'} 상세 보기`}
          aria-pressed={active} onClick={() => setSelected(name)} onFocus={() => setSelected(name)}
          onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelected(name); } }}>
          <circle cx={x(p.intensity)} cy={y(p.ceiling)} r="18" className={active ? 'point-glow active' : 'point-glow'} />
          <circle cx={x(p.intensity)} cy={y(p.ceiling)} r="6" className="point-core" />
          <text x={x(p.intensity)} y={y(p.ceiling) + (name === 'prefill' ? -17 : 29)} textAnchor="middle" className="point-label">{name === 'prefill' ? 'PREFILL' : 'DECODE'}</text>
        </g>;
      })}
      <text x="72" y="15" className="axis-title">TFLOP/s · 상한</text><text x="393" y="328" textAnchor="middle" className="axis-title">Arithmetic intensity · FLOP/byte →</text>
    </svg>
    <div className={`chart-insight ${phaseName.toLowerCase()}`}>
      <span className="insight-symbol">↗</span>
      <div><strong>{phaseName} · {phase.bound === 'memory' ? 'Memory-bound' : 'Compute-bound'}</strong><p>{phase.bound === 'memory' ? '이 설정에서는 메모리 이동이 상한을 정합니다. 읽는 바이트나 대역폭을 바꿔 보세요.' : '이 설정에서는 연산량이 상한을 정합니다. 연산 능력을 바꿔 변화를 확인해 보세요.'}</p></div>
      <div className="insight-value"><strong>{number(phase.intensity, 2)}</strong><span>FLOP / byte</span></div>
    </div>
  </>;
}

export function Traffic({ phase, name }: { phase: Phase; name: string }) {
  const parts = [
    { label: 'Weights', key: 'weights', value: phase.traffic.weights, color: '#9aa9c4' },
    { label: 'KV read', key: 'kv_read', value: phase.traffic.kv_read, color: '#e8b571' },
    { label: 'KV write', key: 'kv_write', value: phase.traffic.kv_write, color: '#af885c' },
    { label: 'Activations', key: 'activations', value: phase.traffic.activations, color: '#75cfc5' },
  ];
  return <div className="traffic" data-testid="traffic">
    <div className="traffic-total"><span>{name} 한 스텝의 전송량</span><strong>{memory(phase.traffic.total)}</strong></div>
    <div className="traffic-bar" role="img" aria-label={`${name} 메모리 전송량 ${memory(phase.traffic.total)}`}>
      {parts.map(p => <div key={p.key} title={`${p.label}: ${memory(p.value)}`} style={{ width: `${p.value / phase.traffic.total * 100}%`, background: p.color }} />)}
    </div>
    <div className="traffic-rows">{parts.map(p => <div key={p.key}><span><i className="dot" style={{ background: p.color }} />{p.label}</span><strong>{memory(p.value)}</strong><small>{number(p.value / phase.traffic.total * 100)}%</small></div>)}</div>
    <div className="cost-pair"><div><span>연산에 필요한 시간</span><strong>{ms(phase.computeMs)} <small>ms</small></strong></div><div><span>전송에 필요한 시간</span><strong>{ms(phase.memoryMs)} <small>ms</small></strong></div></div>
    <p className="footnote">둘 중 긴 시간이 이 모델의 하한입니다. 전송량은 resident memory와 다릅니다.</p>
  </div>;
}

export function Timeline({ result, settings }: { result: Analysis; settings: Settings }) {
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => { setReduced(media.matches); if (media.matches) setPlaying(false); };
    update(); media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);
  useEffect(() => { setPlaying(false); setProgress(0); }, [settings]);
  useEffect(() => {
    if (!playing || reduced) return;
    const timer = window.setInterval(() => setProgress(p => Math.min(settings.output, p + 1)), 100);
    return () => window.clearInterval(timer);
  }, [playing, settings.output, reduced]);
  useEffect(() => { if (progress >= settings.output) setPlaying(false); }, [progress, settings.output]);
  const slots = Math.min(28, settings.output);
  const elapsed = progress === 0 ? 0 : result.prefill.ms + result.decodeTimes.slice(0, progress - 1).reduce((a, b) => a + b, 0);
  return <>
    <div className="timeline-track">
      <div className={`prompt-block ${progress > 0 ? 'complete' : ''}`}><span>PROMPT</span><strong>{number(settings.prompt, 0)}</strong><small>tokens → Prefill</small></div>
      <span className="flow-arrow">→</span>
      <div className="token-sequence">{Array.from({ length: slots }, (_, i) => {
        const token = i === 0 ? 1 : 1 + Math.ceil(i * (settings.output - 1) / (slots - 1));
        return <div key={i} className={`token ${token <= progress ? 'filled' : ''} ${i === 0 ? 'first' : ''}`} title={`출력 토큰 ${token}`}><span>{i === 0 ? '01' : i === slots - 1 ? number(settings.output, 0) : '·'}</span></div>;
      })}<div className="sequence-labels"><span>첫 토큰 · TTFT</span><span>순차 생성 · Decode →</span></div></div>
    </div>
    <div className="timeline-controls"><button className="play-button" aria-label={reduced ? '다음 토큰' : playing ? '일시정지' : '타임라인 재생'} onClick={() => {
      if (reduced) setProgress(p => p >= settings.output ? 0 : p + 1);
      else { if (progress >= settings.output) setProgress(0); setPlaying(!playing); }
    }}>{reduced ? '→' : playing ? 'Ⅱ' : '▶'}</button><input aria-label="생성 토큰 위치" type="range" min="0" max={settings.output} value={progress} onChange={e => { setPlaying(false); setProgress(Number(e.target.value)); }} /><span className="timeline-position">{progress} / {settings.output}</span><span className="timeline-time">{ms(elapsed)} ms <small>계산값</small></span></div>
    <div className="latency-equation"><span><i className="dot cyan" />TTFT <strong>{ms(result.prefill.ms)} ms</strong></span><b>+</b><span><i className="dot amber" />Decode 합계 <strong>{ms(result.decodeMs)} ms</strong></span><b>=</b><span>전체 <strong>{ms(result.totalMs)} ms</strong></span></div>
    <p className="footnote">재생 속도와 토큰 칸은 이해를 위한 도식입니다. 실제 시간 비율이 아니며, 긴 출력은 묶어서 표시합니다.</p>
  </>;
}
