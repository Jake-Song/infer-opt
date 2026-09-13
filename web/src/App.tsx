import { useEffect, useMemo, useState } from 'react';
import { analyze, DEFAULTS, EXPERIMENTS, HIDDENS, memory, ms, number } from './model';
import type { Precision, Settings } from './model';
import { Roofline, Timeline, Traffic } from './Charts';

function Icon({ name }: { name: 'sliders' | 'chart' | 'layers' | 'arrow' | 'reset' | 'pin' }) {
  const paths = {
    sliders: <><path d="M4 6h16M4 12h16M4 18h16" /><path d="M8 3v6M16 9v6M10 15v6" /></>,
    chart: <><path d="M4 4v16h16M7 15l5-5 4 2 4-7" /></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 16l9 5 9-5" /></>,
    arrow: <><path d="M5 12h14m-6-6 6 6-6 6" /></>,
    reset: <><path d="M4 10a8 8 0 1 1 1 8M4 4v6h6" /></>,
    pin: <><path d="m9 3 6 0-1 6 4 4v2H6v-2l4-4-1-6ZM12 15v6" /></>,
  };
  return <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function Slider({ label, value, min, max, onChange, unit, marks }: {
  label: string; value: number; min: number; max: number; onChange: (v: number) => void; unit: string; marks: string[];
}) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  const valid = draft.trim() !== '' && Number.isInteger(Number(draft)) && Number(draft) >= min && Number(draft) <= max;
  return <div className="slider-control"><div className="control-label"><span>{label}</span><span><input aria-label={`${label} 값`} type="number" min={min} max={max} step="1" value={draft} aria-invalid={!valid} onBlur={() => { if (!valid) setDraft(String(value)); }} onChange={e => { setDraft(e.target.value); const v = Number(e.target.value); if (e.target.value.trim() && Number.isInteger(v) && v >= min && v <= max) onChange(v); }} /><small>{unit}</small></span></div><input type="range" aria-label={label} min={min} max={max} value={value} onChange={e => { setDraft(e.target.value); onChange(Number(e.target.value)); }} style={{ '--fill': `${(value - min) / (max - min) * 100}%` } as React.CSSProperties} /><div className="range-labels">{marks.map(m => <span key={m}>{m}</span>)}</div>{!valid && <p className="input-error">{min}–{number(max, 0)} 정수를 입력하세요. 마지막 유효값을 사용합니다.</p>}</div>;
}

function HardwareInput({ label, value, min, max, unit, onChange }: {
  label: string; value: number; min: number; max: number; unit: string; onChange: (v: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));
  const valid = draft.trim() !== '' && Number.isFinite(Number(draft)) && Number(draft) >= min && Number(draft) <= max;
  return <div className="hardware-field"><label>{label}<span className={`input-box ${!valid ? 'invalid' : ''}`}><input type="number" min={min} max={max} step="any" value={draft} aria-invalid={!valid} onChange={e => { setDraft(e.target.value); const v = Number(e.target.value); if (e.target.value.trim() && Number.isFinite(v) && v >= min && v <= max) onChange(v); }} /><span>{unit}</span></span></label>{!valid && <p className="input-error" role="alert">{min}–{number(max, 0)} 값을 입력하세요. 마지막 유효값을 사용합니다.</p>}</div>;
}

export default function App() {
  const [settings, setSettings] = useState<Settings>({ ...DEFAULTS });
  const [baselineSettings, setBaselineSettings] = useState<Settings | null>(null);
  const [activeExperiment, setActiveExperiment] = useState<string | null>(null);
  const [trafficPhase, setTrafficPhase] = useState<'prefill' | 'decode'>('decode');
  const [hardwareRevision, setHardwareRevision] = useState(0);
  const result = useMemo(() => analyze(settings), [settings]);
  const baseline = useMemo(() => baselineSettings ? analyze(baselineSettings) : null, [baselineSettings]);
  function change<K extends keyof Settings>(key: K, value: Settings[K]) {
    setSettings(previous => ({ ...previous, [key]: value }));
  }
  const reset = () => { setSettings({ ...DEFAULTS }); setBaselineSettings(null); setActiveExperiment(null); setTrafficPhase('decode'); setHardwareRevision(v => v + 1); };
  const phase = trafficPhase === 'decode' && result.decode ? result.decode : result.prefill;
  const phaseName = trafficPhase === 'decode' && result.decode ? 'Decode' : 'Prefill';
  const metrics = [
    { label: '첫 토큰까지', name: 'TTFT', value: ms(result.prefill.ms), unit: 'ms', old: baseline?.prefill.ms, current: result.prefill.ms, detail: 'Prefill · 첫 출력 토큰 포함', tone: 'cyan' },
    { label: '토큰 하나마다', name: 'AVG. TPOT', value: result.tpot === null ? '해당 없음' : ms(result.tpot), unit: result.tpot === null ? '' : 'ms', old: baseline?.tpot, current: result.tpot, detail: '증가하는 문맥의 Decode 평균', tone: 'amber' },
    { label: '요청 완료까지', name: 'TOTAL LATENCY', value: ms(result.totalMs), unit: 'ms', old: baseline?.totalMs, current: result.totalMs, detail: `${settings.output}개 출력 토큰 / 요청`, tone: 'neutral' },
    { label: '배치 전체 처리량', name: 'OUTPUT THROUGHPUT', value: number(result.throughput, 0), unit: 'tok/s', old: baseline?.throughput, current: result.throughput, detail: 'Batch × 출력 토큰 ÷ 전체 시간', tone: 'neutral' },
  ];
  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="#"><span className="brand-mark"><i /><i /><i /></span><span>inference<span className="brand-light">lab</span><span className="brand-period">.</span></span></a><nav aria-label="주 메뉴"><a className="nav-active" href="#playground">Playground</a><a href="#experiments">실험 가이드 <span>↗</span></a><a href="#methodology">계산 모델</a></nav><span className="local-status"><i />LOCAL LEARNING LAB</span></header>
    <main id="playground">
      <div className="page-intro"><div><div className="eyebrow"><span>01 / FOUNDATIONS</span><span className="eyebrow-line" />INFERENCE PERFORMANCE</div><h1>추론 성능, <span>직접 움직여 보세요.</span></h1><p>같은 모델, 다른 병목. 워크로드를 바꾸며 GPU가 어디서 기다리는지 살펴보세요.</p></div><div className="intro-label"><span className="pulse-dot" />ANALYTIC ESTIMATES<span>실측이 아닌, 이해를 위한 계산 모델</span></div></div>
      <div className="workspace">
        <aside className="controls panel" aria-label="워크로드 설정">
          <div className="panel-heading"><h2><Icon name="sliders" />워크로드 설정</h2><button className="icon-button" aria-label="초기화" title="초기화" onClick={reset}><Icon name="reset" /></button></div>
          <div className="control-section"><div className="section-label">01 <span>WORKLOAD</span></div>
            <Slider label="Batch size" value={settings.batch} min={1} max={128} onChange={v => change('batch', v)} unit="seq" marks={['1', '64', '128']} />
            <Slider label="Prompt length" value={settings.prompt} min={16} max={32768} onChange={v => change('prompt', v)} unit="tok" marks={['16', '16k', '32k']} />
            <Slider label="Output length" value={settings.output} min={1} max={512} onChange={v => change('output', v)} unit="tok" marks={['1', '256', '512']} />
          </div>
          <div className="control-section"><div className="section-label">02 <span>MODEL</span><span className="tiny-badge">DECODER-ONLY</span></div>
            <label className="select-label">Hidden dimension<select aria-label="Hidden dimension" value={settings.hidden} onChange={e => change('hidden', Number(e.target.value))}>{HIDDENS.map(h => <option key={h} value={h}>{number(h, 0)}</option>)}</select></label>
            <div className="precision-label">Precision</div><div className="segmented precision" role="group" aria-label="Precision">{(['float32', 'float16', 'bfloat16'] as Precision[]).map((p, i) => <button key={p} aria-pressed={settings.precision === p} onClick={() => change('precision', p)}>{['FP32', 'FP16', 'BF16'][i]}</button>)}</div>
            <div className="model-spec"><span>16 layers</span><i />16:4 GQA<i /><span>{number(result.parameters / 1e6)}M params</span></div>
          </div>
          <div className="control-section hardware"><div className="section-label">03 <span>HARDWARE</span><span className="tiny-badge">가상 프로필</span></div>
            <HardwareInput key={`compute-${hardwareRevision}`} label="연산 능력" value={settings.compute} min={0.1} max={10000} unit="TFLOP/s" onChange={v => change('compute', v)} />
            <HardwareInput key={`bandwidth-${hardwareRevision}`} label="메모리 대역폭" value={settings.bandwidth} min={1} max={100000} unit="GB/s" onChange={v => change('bandwidth', v)} />
            <p className="footnote">특정 GPU의 스펙이 아닙니다. Precision을 바꿨다면 해당 정밀도의 연산 능력을 직접 입력하세요.</p>
          </div>
          <div className="controls-footer"><button className="pin-button" onClick={() => { setBaselineSettings({ ...settings }); setActiveExperiment(null); }}><Icon name="pin" />기준 저장 <span>+</span></button><p>현재 설정을 고정하고 변화를 비교하세요.</p></div>
        </aside>
        <div className="results-column">
          <div className="metrics">{metrics.map(m => <article className={`metric-card ${m.tone}`} key={m.name}><div className="metric-title">{m.label}<span>{m.name}</span></div><div className={`metric-value ${m.current === null ? 'empty' : ''}`} data-testid={m.name}><strong>{m.value}</strong><span>{m.unit}</span></div><div className="metric-detail">{m.detail}</div>{baseline && <div className="metric-comparison">기준 대비 {m.old != null && m.current != null ? `${m.current >= m.old ? '+' : ''}${number((m.current / m.old - 1) * 100)}%` : '해당 없음'}</div>}</article>)}</div>
          {baselineSettings && <div className="baseline-strip" role="status"><Icon name="pin" /><span>비교 기준 <strong>B{baselineSettings.batch} · {number(baselineSettings.prompt)} prompt · {baselineSettings.output} output · H{baselineSettings.hidden} · {baselineSettings.precision} · {baselineSettings.compute} TFLOP/s · {baselineSettings.bandwidth} GB/s</strong></span><button onClick={() => { setBaselineSettings(null); setActiveExperiment(null); }} aria-label="비교 기준 제거">×</button></div>}
          <section className="panel roofline-panel"><div className="panel-heading"><div><div className="section-kicker">THE PERFORMANCE CEILING</div><h2>어디서 병목이 생길까요?</h2></div><span className="outlined-tag">Roofline model <Icon name="chart" /></span></div><Roofline result={result} settings={settings} baseline={baseline} baselineSettings={baselineSettings} /></section>
          <div className="lower-grid"><section className="panel timeline-panel"><div className="panel-heading"><div><div className="section-kicker">FROM PROMPT TO TOKENS</div><h2>한 번에 읽고, 하나씩 생성합니다.</h2></div><span className="number-tag">01 → N</span></div><Timeline result={result} settings={settings} /></section>
            <section className="panel memory-panel"><div className="panel-heading"><h2><Icon name="layers" />메모리 트래픽</h2><div className="segmented phase-tabs" role="group" aria-label="메모리 단계"><button aria-pressed={phaseName === 'Prefill'} onClick={() => setTrafficPhase('prefill')}>Prefill</button><button disabled={!result.decode} aria-pressed={phaseName === 'Decode'} onClick={() => setTrafficPhase('decode')}>Decode</button></div></div><Traffic phase={phase} name={phaseName} /></section></div>
          <div className="resident-strip"><span><Icon name="layers" />상주 메모리 <small>Resident</small></span><span>Weights <strong>{memory(result.weights)}</strong></span><span>최종 KV <strong>{memory(result.residentKv)}</strong></span><span className="resident-note">사용된 KV 기준 · 활성값 / 런타임 여유 공간 제외</span></div>
        </div>
      </div>
      <section className="experiments-section" id="experiments"><div className="section-header"><div><div className="section-kicker">LEARN BY CHANGING ONE THING</div><h2>작은 변화에서 시작하는 실험</h2></div><p>카드를 선택하면 기준과 실험 설정이 함께 적용됩니다.</p></div><div className="experiment-grid">{EXPERIMENTS.map((e, i) => <button key={e.id} className={`experiment-card ${activeExperiment === e.id ? 'selected' : ''}`} aria-pressed={activeExperiment === e.id} onClick={() => { setBaselineSettings({ ...e.before }); setSettings({ ...e.after }); setActiveExperiment(e.id); setHardwareRevision(v => v + 1); }}><div className="experiment-top"><span className="experiment-number">0{i + 1}</span><span>{e.tag}</span><Icon name="arrow" /></div><h3>{e.label}</h3><p>{e.description}</p><span className="experiment-action">{activeExperiment === e.id ? '실험 적용됨' : '실험 시작하기'} <span>↗</span></span></button>)}</div></section>
      <section className="methodology panel" id="methodology"><div className="methodology-intro"><div className="section-kicker">BEHIND THE NUMBERS</div><h2>숫자를 읽는 방법</h2><p>예측을 세우고, 실제 하드웨어에서 검증하세요.</p></div><div className="methodology-content"><div className="formula-row"><code>AI = FLOPs / Bytes</code><code>t ≈ max(FLOPs / π, Bytes / β)</code><code>Roofline = min(π, β × AI)</code></div><details><summary>계산 가정과 한계 알아보기 <span>+</span></summary><div className="methodology-details"><p>모든 성능 수치는 계산 모델의 하한 시간 / 상한 처리량이며 실측이 아닙니다. Roofline 위 점도 달성 성능이 아닌 모델의 상한입니다. 실제 TTFT에는 큐 대기, 토큰화, 네트워크 등이 추가됩니다.</p><p>FLOPs는 multiply-add를 2회로 계산하고, causal attention에 필요한 쌍만 셉니다. 메모리 트래픽은 가중치를 매 스텝 한 번 읽고, KV 읽기·쓰기와 비융합 baseline의 중간 활성값 및 logits를 합산한 근사입니다. 캐시 재사용과 fusion에 따라 실제 트래픽은 달라집니다. Kernel launch와 낮은 활용률은 모델에 포함하지 않습니다.</p><p>Prefill이 첫 출력 토큰을 만듭니다. 나머지 N−1개는 문맥을 한 칸씩 늘려 계산하므로 총 지연 = TTFT + 각 decode 시간의 합입니다. TPOT은 이 N−1개 스텝의 평균이며 roofline의 Decode 점과 메모리 트래픽은 첫 decode 스텝 기준입니다.</p><p>FP16과 BF16은 모두 2 bytes, FP32는 4 bytes입니다. 연산 능력은 독립 입력값입니다. KV 상주량은 prompt + output − 1 위치의 사용량이며 실제 사전할당 용량, 활성값, 런타임 메모리를 포함하지 않습니다. GB는 10⁹ bytes입니다.</p><p>모델은 16 layers, 16 query heads / 4 KV heads, vocab 32,000, untied embedding / LM head, intermediate = hidden × 11/4를 사용합니다. 긴 문맥은 분석용 확장이며 기존 노트북 실행 용량이나 모델 품질을 보장하지 않습니다.</p><p>계산식: <code>infer_opt/profiling.py</code> · 읽을거리: <a href="https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html" target="_blank" rel="noreferrer">NVIDIA GPU Performance Background ↗</a></p></div></details></div></section>
      <footer><span className="footer-brand">inference lab<span>.</span></span><span>가설을 세우고. 병목을 찾고. 측정으로 확인하세요.</span><span>BUILT FOR CURIOSITY / 01</span></footer>
    </main>
  </div>;
}
