import {useEffect, useRef, useState} from 'react';
import {Activity, ArrowDownToLine, ArrowRight, Check, ChevronRight, CircleHelp, DoorOpen, FileSpreadsheet, LayoutDashboard, LoaderCircle, Radio, ShieldCheck, TrainFront, UploadCloud, Waves, Wind, X} from 'lucide-react';
import {ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar, ReferenceArea} from 'recharts';

type Subsystem = 'door'|'acv'|'rail'|'shm';
type Row = Record<string, string|number>;
type Result = {subsystem:Subsystem; file_id:string; model:string; rows:Row[]; csv:string; csv_filename:string; seconds:number; warnings:string[]; details: {segments?:number; abnormal?:number; timeline?:{time:string;elapsed:number;current:number;position:number}[]; segment_evidence?:{start:number;end:number;label:string;support:number}[]; model_support?:number; temperature_trend?:Record<string,number|null>[]; vibration_preview?:{time:number;side_i:number;side_ii:number}[]; spectrum?:{frequency:number;side_i:number;side_ii:number}[]; ranking?:{car:string;score:number;mean_temperature_residual:number}[]; heatmap?:number[][]; speed_mps?:number; histogram?:number[]; bin_edges?:number[]; stress_preview?:number[]; explanation:string; evidence?:{feature:string;value:number;reference:number;standardised_deviation:number}[]}};
type Model = {ready:boolean; model?:string; validation_score?:number; result_type?:string; validation?:string};
const configs = {
  door:{name:'Door systems',short:'Door',icon:DoorOpen,label:'OPERATION DETECTION',description:'Find every opening and closing cycle. Identify abnormal resistance before it becomes a disruption.',file:'Continuous sensor stream · CSV',metric:'IoU-weighted F1',color:'#d77b4c'},
  acv:{name:'Air conditioning',short:'ACV',icon:Wind,label:'FAULT LOCALISATION',description:'Compare the train’s eight cars and rank the most likely source of a refrigerant leak.',file:'Eight-car telemetry · XLSX',metric:'Rank-decay score',color:'#537e85'},
  rail:{name:'Rail corrugation',short:'Rail',icon:Waves,label:'SIDE CLASSIFICATION',description:'Read the vibration signature. Locate corrugation on Side I or Side II of the track.',file:'10 kHz vibration recording · CSV',metric:'Macro F1',color:'#728764'},
  shm:{name:'Structural health',short:'SHM',icon:Activity,label:'FATIGUE ESTIMATION',description:'Turn a dynamic stress history into an estimate of accumulated fatigue damage.',file:'Headerless stress sequence · CSV',metric:'1 − MAPE',color:'#827090'},
};
const keys = Object.keys(configs) as Subsystem[];

function download(data:BlobPart,name:string,type='text/csv') {
  const url=URL.createObjectURL(new Blob([data],{type})); const a=document.createElement('a'); a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

export default function App(){
  const [selected,setSelected]=useState<Subsystem>('door');
  const [models,setModels]=useState<Partial<Record<Subsystem,Model>>>({});
  const [files,setFiles]=useState<File[]>([]);
  const [results,setResults]=useState<Result[]>([]);
  const [busy,setBusy]=useState(false);
  const [progress,setProgress]=useState('');
  const [error,setError]=useState('');
  const [drag,setDrag]=useState(false);
  const [tab,setTab]=useState<'analyse'|'exports'|'methods'>('analyse');
  const input=useRef<HTMLInputElement>(null);
  const config=configs[selected]; const Icon=config.icon;
  const active=results.filter(r=>r.subsystem===selected);
  const latest=active.at(-1);
  useEffect(()=>{fetch('/api/models').then(r=>{if(!r.ok)throw Error();return r.json();}).then(setModels).catch(()=>setError('Could not connect to the analysis service. Please retry shortly.'));},[]);
  function select(s:Subsystem){setSelected(s);setFiles([]);setError('');setTab('analyse');}
  function choose(chosen:File[]){setError(''); if(selected==='door'&&chosen.length>1){setError('Door analysis takes one continuous stream at a time.');return;} const ext=selected==='acv'?'.xlsx':'.csv';if(chosen.some(f=>!f.name.toLowerCase().endsWith(ext))){setError(`Select ${ext} files for this subsystem.`);return;}setFiles(chosen);}
  async function analyse(){
    setBusy(true);setError('');
    try{for(let i=0;i<files.length;i++){
      setProgress(`Analysing ${i+1} of ${files.length} · ${files[i].name}`);
      const body=new FormData();body.append('file',files[i]);
      const response=await fetch(`/api/analyse/${selected}`,{method:'POST',body});
      const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'The file could not be processed. Check its format.');
      setResults(old=>[...old.filter(r=>!(r.subsystem===selected&&(selected==='door'||r.file_id===data.file_id))),data]);
    }setFiles([]);}catch(e){setError((e as Error).message);}finally{setBusy(false);setProgress('');}
  }
  async function exportZip(){
    setError('');const grouped:Partial<Record<Subsystem,Row[]>>={};for(const r of results)(grouped[r.subsystem]??=[]).push(...r.rows);
    try{const response=await fetch('/api/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(grouped)});if(!response.ok){const d=await response.json();throw Error(d.detail);}download(await response.blob(),'predictions.zip','application/zip');}catch(e){setError((e as Error).message);}
  }
  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e=>{e.preventDefault();setTab('analyse');}}><span className="brand-mark"><TrainFront size={25}/></span><span>doors are<br/><b>closing.</b></span></a>
      <div className="workspace-label">CONDITION MONITORING <span>01</span></div>
      <button className={`nav-item ${tab==='analyse'?'active':''}`} onClick={()=>setTab('analyse')}><LayoutDashboard size={18}/> Analysis workspace</button>
      <div className="nav-label">SUBSYSTEMS</div>
      {keys.map(s=>{const I=configs[s].icon;return <button disabled={busy} className={`sub-nav ${selected===s&&tab==='analyse'?'selected':''}`} key={s} onClick={()=>select(s)}><I size={18}/><span>{configs[s].name}</span><span className={`status-dot ${models[s]?.ready?'ready':''}`}/></button>;})}
      <div className="nav-divider"/>
      <button className={`nav-item ${tab==='exports'?'active':''}`} onClick={()=>setTab('exports')}><ArrowDownToLine size={18}/> Prediction exports <span className="count">{results.length}</span></button>
      <button className={`nav-item ${tab==='methods'?'active':''}`} onClick={()=>setTab('methods')}><ShieldCheck size={18}/> Models & validation</button>
      <div className="sidebar-bottom"><div className="live-label"><span className="status-dot ready"/> NEBULA X · 2026</div><p>Better signals.<br/>More reliable journeys.</p><span className="sidebar-foot">LTA TRAIN CONDITION MONITORING</span></div>
    </aside>
    <div className="main-shell">
      <header className="topbar"><div>Workspace <ChevronRight size={13}/> <b>{tab==='analyse'?'Subsystem analysis':tab==='exports'?'Prediction exports':'Models & validation'}</b></div><span className="system-status"><span className={`status-dot ${keys.every(k=>models[k]?.ready)?'ready':''}`}/>{keys.every(k=>models[k]?.ready)?'All models ready':'Connecting to models'}</span></header>
      <main>
        <nav className="mobile-nav" aria-label="Workspace pages">{(['analyse','exports','methods'] as const).map(t=><button key={t} className={tab===t?'active':''} onClick={()=>setTab(t)}>{t==='analyse'?'Analysis':t==='exports'?'Exports':'Validation'}</button>)}</nav>
        <div className="page-heading"><div><div className="eyebrow">RAIL INTELLIGENCE, MADE CLEAR</div><h1>{tab==='analyse'?'Every signal tells a story.':tab==='exports'?'Ready for the next step.':'Evidence behind the prediction.'}</h1><p>{tab==='analyse'?'From onboard data to a clearer maintenance decision.':tab==='exports'?'Validated outputs, in the exact format your submission needs.':'Measured local results and the assumptions that shape them.'}</p></div><div className="edition">NEBULA X<br/><b>PS3 / 2026</b></div></div>
        {error&&<div className="error" role="alert"><CircleHelp size={18}/>{error}<button aria-label="Dismiss error" onClick={()=>setError('')}><X size={16}/></button></div>}
        {tab==='analyse'&&<>
          <div className="system-cards">{keys.map((s,i)=>{const c=configs[s],I=c.icon;return <button disabled={busy} key={s} onClick={()=>select(s)} className={`system-card ${selected===s?'chosen':''}`} style={{'--accent':c.color} as React.CSSProperties}><div className="card-top"><span className="card-icon"><I size={22}/></span><span className="card-number">0{i+1}</span></div><h3>{c.name}</h3><p>{c.label}</p><div className="card-bottom"><span>{models[s]?.ready?'Available for analysis':'Model loading'}</span>{selected===s?<span className="selected-check"><Check size={12}/></span>:<ArrowRight size={15}/>}</div></button>;})}</div>
          <div className="analysis-grid">
            <section className="panel upload-panel"><div className="section-header"><span className="step-number">01</span><div><h2>Upload your data</h2><p>{config.file}</p></div><Icon size={22}/></div>
              <div role="button" tabIndex={busy?-1:0} aria-label="Choose input files" className={`dropzone ${drag?'dragging':''}`} onClick={()=>!busy&&input.current?.click()} onKeyDown={e=>{if((e.key==='Enter'||e.key===' ')&&!busy)input.current?.click();}} onDragOver={e=>{e.preventDefault();if(!busy)setDrag(true);}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);if(!busy)choose(Array.from(e.dataTransfer.files));}}>
                <input ref={input} type="file" aria-label="Upload data" accept={selected==='acv'?'.xlsx':'.csv'} multiple={selected!=='door'} disabled={busy} onChange={e=>{choose(Array.from(e.target.files??[]));e.currentTarget.value="";}}/>
                <span className="upload-icon"><UploadCloud size={30}/></span><h3>{files.length?`${files.length} file${files.length>1?'s':''} selected`:'Drop a file. Find the signal.'}</h3><p>{files.length?files.map(f=>f.name).slice(0,3).join(', '):'Drag your data here, or browse files'}</p><span className="file-pill">{selected==='acv'?'XLSX':'CSV'} <span>·</span> UP TO 40 MB PER FILE</span>
              </div>
              <div className="input-note"><ShieldCheck size={16}/><span>Files are checked before analysis and removed after processing.</span></div>
              <button className="primary-button" disabled={!files.length||busy||!models[selected]?.ready} onClick={analyse}>{busy?<><LoaderCircle className="spin" size={18}/>{progress}</>:<>Analyse {config.short} data<ArrowRight size={18}/></>}</button>
            </section>
            <section className="context-panel"><div className="eyebrow">WHAT WE LOOK FOR</div><h2>{config.name}<span className="accent-dot">.</span></h2><p>{config.description}</p><div className="rail-illustration" aria-hidden="true"><div className="track-line"/>{Array.from({length:8},(_,i)=><div key={i} className={`train-car ${i===2?'highlight':''}`}><span>{String(i+1).padStart(2,'0')}</span><div/><div/></div>)}<div className="track-line bottom"/></div><div className="context-facts"><div><span>OUTPUT</span><b>{selected==='door'?'Timed operation segments':selected==='acv'?'Complete eight-car ranking':selected==='rail'?'Normal / Side I / Side II':'Cumulative fatigue damage'}</b></div><div><span>VALIDATION METRIC</span><b>{config.metric}</b></div></div><button className="text-button" onClick={()=>setTab('methods')}>How this analysis works <ArrowRight size={14}/></button></section>
          </div>
          <section className="panel results-panel"><div className="section-header"><span className="step-number">02</span><div><h2>Understand the result</h2><p>{latest?`${latest.file_id} · processed in ${latest.seconds.toFixed(2)}s`:'Your prediction and the evidence behind it will appear here.'}</p></div>{latest&&<button className="secondary-button" onClick={()=>download(latest.csv,latest.csv_filename)}><ArrowDownToLine size={15}/>Download CSV</button>}</div>
            {latest?<ResultView result={latest}/>:<div className="empty-result"><span className="empty-wave"><Radio size={30}/></span><div><h3>Listening for your first signal</h3><p>Upload {config.short} data to get a prediction, visual evidence, and a submission-ready CSV.</p></div><span className="awaiting">AWAITING DATA</span></div>}
            {active.length>1&&<div className="batch-note">{active.length} {config.short} files analysed this session. All are included in Prediction exports.</div>}
          </section>
        </>}
        {tab==='exports'&&<section className="panel export-panel"><div className="section-header"><ArrowDownToLine size={22}/><div><h2>Submission package</h2><p>Only the required prediction CSVs. No raw data or subfolders.</p></div><button className="primary-button compact" disabled={!results.length} onClick={exportZip}>Download predictions.zip<ArrowDownToLine size={16}/></button></div>{!results.length?<div className="empty-result">Analyse a file to add its predictions to your package.</div>:<div className="export-list">{keys.map(s=>{const rs=results.filter(r=>r.subsystem===s);if(!rs.length)return null;return <div key={s}><FileSpreadsheet size={26}/><span><b>{s}_predictions.csv</b><small>{rs.length} source file(s) · {rs.reduce((n,r)=>n+r.rows.length,0)} predictions</small></span><Check size={18}/><button className="text-button" onClick={()=>setResults(old=>old.filter(r=>r.subsystem!==s))}>Remove</button></div>;})}</div>}</section>}
        {tab==='methods'&&<div className="method-grid">{keys.map(s=><section key={s} className="panel method-card"><div className="eyebrow">{configs[s].label}</div><h2>{configs[s].name}</h2><div className="score">{models[s]?.validation_score?.toFixed(4)??'—'}<span>{configs[s].metric}</span></div><p>{models[s]?.validation??'Benchmark results are being prepared.'}</p><div className="method-foot"><b>{models[s]?.model?.replaceAll('_',' ')??'Pending'}</b><span>LOCAL VALIDATION</span></div></section>)}<div className="method-notice"><ShieldCheck size={20}/><p>These are internal validation results, not organiser test scores. Validation uses complete segments or files. ACV has only six labelled cases. The final models are refitted on all labelled training data after selection.</p></div></div>}
        <footer><span><TrainFront size={14}/> Doors Are Closing</span><span>Built for the journeys ahead.</span><span>NEBULA X × LTA · 2026</span></footer>
      </main>
    </div>
  </div>;
}

function ResultView({result:r}:{result:Result}){
  const d=r.details;
  return <div className="result-content">
    <div className="result-summary"><span className="result-icon"><Check size={22}/></span><div><div className="eyebrow">ANALYSIS COMPLETE</div><h2>{r.subsystem==='door'?`${d.segments} operations · ${d.abnormal} abnormal`:r.subsystem==='acv'?`Inspect Car ${d.ranking?.[0].car} first`:r.subsystem==='rail'?String(r.rows[0].prediction):`Damage estimate: ${Number(r.rows[0].prediction).toFixed(5)}`}</h2></div><span className="model-tag">{r.model.replaceAll('_',' ')}</span></div>
    <p className="explanation">{d.explanation}</p>
    {d.timeline&&<div className="chart"><h3>Recorded timeline · elapsed seconds · orange regions: abnormal operations</h3><ResponsiveContainer width="100%" height={220}><LineChart data={d.timeline}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="elapsed" type="number" domain={['dataMin','dataMax']}/><YAxis width={48}/><YAxis yAxisId="position" orientation="right" width={48}/><Tooltip/>{d.segment_evidence?.filter(s=>s.label!=='Normal').map((s,i)=><ReferenceArea key={i} x1={s.start} x2={s.end} fill="#d87949" fillOpacity={.2}/>)}<Line name="Motor current (mA)" dataKey="current" stroke="#d87949" dot={false} strokeWidth={2}/><Line yAxisId="position" name="Door position (encoder)" dataKey="position" stroke="#537e85" dot={false} strokeWidth={1.5}/></LineChart></ResponsiveContainer><p>Downsampled overview; connecting lines span acquisition gaps.</p></div>}
    {d.segment_evidence&&<details className="evidence"><summary>Segment model support <span>Uncalibrated scores, not fault probabilities</span></summary><div className="table-scroll"><table><thead><tr><th>Elapsed start / end (s)</th><th>Class</th><th>Model support</th></tr></thead><tbody>{d.segment_evidence.map((s,i)=><tr key={i}><td>{s.start.toFixed(2)} / {s.end.toFixed(2)}</td><td>{s.label}</td><td>{(s.support*100).toFixed(1)}%</td></tr>)}</tbody></table></div></details>}
    {d.model_support!==undefined&&<p className="explanation">Model support: {(d.model_support*100).toFixed(1)}% — uncalibrated; not a maintenance risk probability.</p>}
    {d.temperature_trend&&<div className="chart"><h3>Cabin temperature by sample · °C · downsampled</h3><ResponsiveContainer width="100%" height={220}><LineChart data={d.temperature_trend}><XAxis dataKey="sample"/><YAxis domain={['auto','auto']} width={48}/><Tooltip/>{d.ranking?.map((c,i)=><Line key={c.car} name={`Car ${c.car}`} dataKey={c.car} stroke={['#d87949','#537e85','#827090','#728764','#b58a46','#65759b','#a06b76','#778b91'][i]} dot={false}/>)}</LineChart></ResponsiveContainer></div>}
    {d.vibration_preview&&<div className="chart"><h3>Side-mean vibration · m/s² · downsampled overview, not a spectral estimate</h3><ResponsiveContainer width="100%" height={180}><LineChart data={d.vibration_preview}><XAxis dataKey="time" hide/><YAxis width={48}/><Tooltip/><Line dataKey="side_i" name="Side I" stroke="#d87949" dot={false}/><Line dataKey="side_ii" name="Side II" stroke="#537e85" dot={false}/></LineChart></ResponsiveContainer></div>}
    {d.spectrum&&<div className="chart"><h3>Mean vibration power spectral density · frequency (Hz)</h3><ResponsiveContainer width="100%" height={180}><LineChart data={d.spectrum}><XAxis dataKey="frequency" type="number"/><YAxis width={48}/><Tooltip/><Line dataKey="side_i" name="Side I" stroke="#d87949" dot={false}/><Line dataKey="side_ii" name="Side II" stroke="#537e85" dot={false}/></LineChart></ResponsiveContainer></div>}
    {d.ranking&&<div className="ranking">{d.ranking.map((car,i)=><div key={car.car}><span className="rank">{i+1}</span><b>Car {car.car}</b><div className="rank-bar"><span style={{width:`${Math.max(2,100*car.score/Math.max(1e-6,...d.ranking!.map(c=>c.score)))}%`}}/></div><span>{car.score.toFixed(2)}</span></div>)}</div>}
    {d.heatmap&&<div className="heatmap"><div className="heatmap-title">Axle-box vibration · RMS (m/s²)<span>Odd positions: Side I · Even: Side II</span></div><div className="heatmap-row"><span/><b>01</b><b>02</b><b>03</b><b>04</b><b>05</b><b>06</b><b>07</b><b>08</b></div>{d.heatmap.map((row,i)=><div className="heatmap-row" key={i}><span>CAR {i+1}</span>{row.map((v,j)=><div key={j} title={`Car ${i+1}, position ${j+1}: ${v.toFixed(3)}`} style={{background:`rgba(215,123,76,${.12+.78*v/Math.max(...d.heatmap!.flat())})`}}>{v.toFixed(2)}</div>)}</div>)}</div>}
    {d.stress_preview&&<div className="chart"><h3>Stress history · downsampled overview (units unspecified)</h3><ResponsiveContainer width="100%" height={180}><LineChart data={d.stress_preview.map((stress,index)=>({stress,index}))}><XAxis dataKey="index" hide/><YAxis width={55}/><Tooltip/><Line dataKey="stress" stroke="#827090" dot={false}/></LineChart></ResponsiveContainer></div>}
    {d.histogram&&<div className="chart"><h3>Rainflow cycles by stress range</h3><ResponsiveContainer width="100%" height={220}><BarChart data={d.histogram.map((n,i)=>({range:d.bin_edges![i].toFixed(1),cycles:n}))}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="range"/><YAxis width={55}/><Tooltip/><Bar dataKey="cycles" fill="#537e85" radius={[3,3,0,0]}/></BarChart></ResponsiveContainer></div>}
    {d.evidence&&<details className="evidence"><summary>Explore feature evidence <span>Descriptive deviations from training data</span></summary><table><thead><tr><th>Signal feature</th><th>Observed</th><th>Reference median</th></tr></thead><tbody>{d.evidence.map(e=><tr key={e.feature}><td>{e.feature.replaceAll('_',' ')}</td><td>{e.value.toPrecision(4)}</td><td>{e.reference.toPrecision(4)}</td></tr>)}</tbody></table></details>}
    <details className="evidence"><summary>Preview prediction rows <span>{r.rows.length} rows</span></summary><div className="table-scroll"><table><thead><tr>{Object.keys(r.rows[0]).map(k=><th key={k}>{k}</th>)}</tr></thead><tbody>{r.rows.slice(0,100).map((row,i)=><tr key={i}>{Object.values(row).map((v,j)=><td key={j}>{String(v)}</td>)}</tr>)}</tbody></table></div></details>
    {r.warnings.map(w=><div className="result-warning" key={w}><CircleHelp size={15}/>{w}</div>)}
  </div>;
}
