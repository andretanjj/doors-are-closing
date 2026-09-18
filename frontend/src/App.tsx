import {useEffect, useRef, useState} from 'react';
import {Activity, ArrowDownToLine, ArrowRight, Check, ChevronRight, CircleHelp, DoorOpen, FileSpreadsheet, LayoutDashboard, LoaderCircle, Radio, ShieldCheck, TrainFront, UploadCloud, Waves, Wind, X} from 'lucide-react';
import {ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar, ReferenceArea} from 'recharts';

type Subsystem = 'door'|'acv'|'rail'|'shm';
type Row = Record<string, string|number>;
type Result = {subsystem:Subsystem; file_id:string; model:string; rows:Row[]; csv:string; csv_filename:string; seconds:number; warnings:string[]; details: {segments?:number; abnormal?:number; timeline?:{time:string;elapsed:number;current:number;position:number}[]; segment_evidence?:{start:number;end:number;label:string;support:number}[]; model_support?:number; temperature_trend?:Record<string,number|null>[]; vibration_preview?:{time:number;side_i:number;side_ii:number}[]; spectrum?:{frequency:number;side_i:number;side_ii:number}[]; ranking?:{car:string;score:number;mean_temperature_residual:number}[]; heatmap?:number[][]; speed_mps?:number; histogram?:number[]; bin_edges?:number[]; stress_preview?:number[]; explanation:string; evidence?:{feature:string;value:number;reference:number;standardised_deviation:number}[]}};
type Model = {ready:boolean; model?:string; validation_score?:number; result_type?:string; validation?:string};
const configs = {
  door:{name:'Door systems',short:'Door',icon:DoorOpen,label:'Door check',description:'Find door cycles and flag resistance.',file:'Sensor stream · CSV',metric:'Match score',color:'#d77b4c'},
  acv:{name:'Air conditioning',short:'ACV',icon:Wind,label:'AC check',description:'Compare eight cars and rank likely leaks.',file:'8-car telemetry · XLSX',metric:'Ranking score',color:'#537e85'},
  rail:{name:'Rail corrugation',short:'Rail',icon:Waves,label:'Rail check',description:'Use vibration to find Side I or II.',file:'10 kHz vibration · CSV',metric:'Class score',color:'#728764'},
  shm:{name:'Structural health',short:'SHM',icon:Activity,label:'Fatigue check',description:'Estimate fatigue from stress data.',file:'Stress sequence · CSV',metric:'Error score',color:'#827090'},
};
const keys = Object.keys(configs) as Subsystem[];
const simpleValidation:Record<Subsystem,string> = {
  door:'Checked on 22 time-ordered samples.',
  acv:'Checked on 6 labelled workbooks.',
  rail:'Checked on 54 files.',
  shm:'Checked on 16 files.',
};

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
  useEffect(()=>{fetch('/api/models').then(r=>{if(!r.ok)throw Error();return r.json();}).then(setModels).catch(()=>setError('Cannot connect to the analysis service.'));},[]);
  function select(s:Subsystem){setSelected(s);setFiles([]);setError('');setTab('analyse');}
  function choose(chosen:File[]){setError(''); if(selected==='door'&&chosen.length>1){setError('Door analysis takes one file.');return;} const ext=selected==='acv'?'.xlsx':'.csv';if(chosen.some(f=>!f.name.toLowerCase().endsWith(ext))){setError(`Select ${ext} files.`);return;}setFiles(chosen);}
  async function analyse(){
    setBusy(true);setError('');
    try{for(let i=0;i<files.length;i++){
      setProgress(`Analysing ${i+1} of ${files.length} · ${files[i].name}`);
      const body=new FormData();body.append('file',files[i]);
      const response=await fetch(`/api/analyse/${selected}`,{method:'POST',body});
      const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'File could not be processed. Check the format.');
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
      <button className={`nav-item ${tab==='analyse'?'active':''}`} onClick={()=>setTab('analyse')}><LayoutDashboard size={18}/> Analysis</button>
      <div className="nav-label">SUBSYSTEMS</div>
      {keys.map(s=>{const I=configs[s].icon;return <button disabled={busy} className={`sub-nav ${selected===s&&tab==='analyse'?'selected':''}`} key={s} onClick={()=>select(s)}><I size={18}/><span>{configs[s].name}</span><span className={`status-dot ${models[s]?.ready?'ready':''}`}/></button>;})}
      <div className="nav-divider"/>
      <button className={`nav-item ${tab==='exports'?'active':''}`} onClick={()=>setTab('exports')}><ArrowDownToLine size={18}/> Exports <span className="count">{results.length}</span></button>
      <button className={`nav-item ${tab==='methods'?'active':''}`} onClick={()=>setTab('methods')}><ShieldCheck size={18}/> Models</button>
      <div className="sidebar-bottom"><div className="live-label"><span className="status-dot ready"/> NEBULA X · 2026</div><p>Better signals.<br/>More reliable journeys.</p><span className="sidebar-foot">LTA TRAIN CONDITION MONITORING</span></div>
    </aside>
    <div className="main-shell">
      <header className="topbar"><div>Workspace <ChevronRight size={13}/> <b>{tab==='analyse'?'Analysis':tab==='exports'?'Exports':'Models'}</b></div><span className="system-status"><span className={`status-dot ${keys.every(k=>models[k]?.ready)?'ready':''}`}/>{keys.every(k=>models[k]?.ready)?'Ready':'Connecting'}</span></header>
      <main>
        <nav className="mobile-nav" aria-label="Workspace pages">{(['analyse','exports','methods'] as const).map(t=><button key={t} className={tab===t?'active':''} onClick={()=>setTab(t)}>{t==='analyse'?'Analysis':t==='exports'?'Exports':'Models'}</button>)}</nav>
        <div className="page-heading"><div><div className="eyebrow">RAIL INTELLIGENCE</div><h1>{tab==='analyse'?'Clearer maintenance decisions.':tab==='exports'?'Download your results.':'Model results.'}</h1><p>{tab==='analyse'?'Upload a file.':tab==='exports'?'Download prediction files.':'Sample scores only.'}</p></div><div className="edition">NEBULA X<br/><b>PS3 / 2026</b></div></div>
        {error&&<div className="error" role="alert"><CircleHelp size={18}/>{error}<button aria-label="Dismiss error" onClick={()=>setError('')}><X size={16}/></button></div>}
        {tab==='analyse'&&<>
          <div className="system-cards">{keys.map((s,i)=>{const c=configs[s],I=c.icon;return <button disabled={busy} key={s} onClick={()=>select(s)} className={`system-card ${selected===s?'chosen':''}`} style={{'--accent':c.color} as React.CSSProperties}><div className="card-top"><span className="card-icon"><I size={22}/></span><span className="card-number">0{i+1}</span></div><h3>{c.name}</h3><p>{c.label}</p><div className="card-bottom"><span>{models[s]?.ready?'Ready':'Loading'}</span>{selected===s?<span className="selected-check"><Check size={12}/></span>:<ArrowRight size={15}/>}</div></button>;})}</div>
          <div className="analysis-grid">
            <section className="panel upload-panel"><div className="section-header"><span className="step-number">01</span><div><h2>Upload your data</h2><p>{config.file}</p></div><Icon size={22}/></div>
              <div role="button" tabIndex={busy?-1:0} aria-label="Choose input files" className={`dropzone ${drag?'dragging':''}`} onClick={()=>!busy&&input.current?.click()} onKeyDown={e=>{if((e.key==='Enter'||e.key===' ')&&!busy)input.current?.click();}} onDragOver={e=>{e.preventDefault();if(!busy)setDrag(true);}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);if(!busy)choose(Array.from(e.dataTransfer.files));}}>
                <input ref={input} type="file" aria-label="Upload data" accept={selected==='acv'?'.xlsx':'.csv'} multiple={selected!=='door'} disabled={busy} onChange={e=>{choose(Array.from(e.target.files??[]));e.currentTarget.value="";}}/>
                <span className="upload-icon"><UploadCloud size={30}/></span><h3>{files.length?`${files.length} file${files.length>1?'s':''} selected`:'Drop a file or browse'}</h3><p>{files.length?files.map(f=>f.name).slice(0,3).join(', '):'Choose your data file'}</p><span className="file-pill">{selected==='acv'?'XLSX':'CSV'} <span>·</span> MAX 40 MB</span>
              </div>
              <div className="input-note"><ShieldCheck size={16}/><span>Deleted after use.</span></div>
              <button className="primary-button" disabled={!files.length||busy||!models[selected]?.ready} onClick={analyse}>{busy?<><LoaderCircle className="spin" size={18}/>{progress}</>:<>Analyse {config.short} data<ArrowRight size={18}/></>}</button>
            </section>
            <section className="context-panel"><div className="eyebrow">LOOKING FOR</div><h2>{config.name}<span className="accent-dot">.</span></h2><p>{config.description}</p><div className="rail-illustration" aria-hidden="true"><div className="track-line"/>{Array.from({length:8},(_,i)=><div key={i} className={`train-car ${i===2?'highlight':''}`}><span>{String(i+1).padStart(2,'0')}</span><div/><div/></div>)}<div className="track-line bottom"/></div><div className="context-facts"><div><span>OUTPUT</span><b>{selected==='door'?'Timed segments':selected==='acv'?'8-car ranking':selected==='rail'?'Normal / Side I / Side II':'Fatigue damage'}</b></div><div><span>SCORE</span><b>{config.metric}</b></div></div><button className="text-button" onClick={()=>setTab('methods')}>How it works <ArrowRight size={14}/></button></section>
          </div>
          <section className="panel results-panel"><div className="section-header"><span className="step-number">02</span><div><h2>Results</h2><p>{latest?`${latest.file_id} · ${latest.seconds.toFixed(2)}s`:'Upload a file to begin.'}</p></div>{latest&&<button className="secondary-button" onClick={()=>download(latest.csv,latest.csv_filename)}><ArrowDownToLine size={15}/>Download CSV</button>}</div>
            {latest?<ResultView result={latest}/>:<div className="empty-result"><span className="empty-wave"><Radio size={30}/></span><div><h3>No result yet</h3><p>Upload data to see the result.</p></div><span className="awaiting">WAITING</span></div>}
            {active.length>1&&<div className="batch-note">{active.length} {config.short} files · included in Exports.</div>}
          </section>
        </>}
        {tab==='exports'&&<section className="panel export-panel"><div className="section-header"><ArrowDownToLine size={22}/><div><h2>Exports</h2><p>Prediction CSVs only.</p></div><button className="primary-button compact" disabled={!results.length} onClick={exportZip}>Download ZIP<ArrowDownToLine size={16}/></button></div>{!results.length?<div className="empty-result">No files yet.</div>:<div className="export-list">{keys.map(s=>{const rs=results.filter(r=>r.subsystem===s);if(!rs.length)return null;return <div key={s}><FileSpreadsheet size={26}/><span><b>{s}_predictions.csv</b><small>{rs.length} file(s) · {rs.reduce((n,r)=>n+r.rows.length,0)} predictions</small></span><Check size={18}/><button className="text-button" onClick={()=>setResults(old=>old.filter(r=>r.subsystem!==s))}>Remove</button></div>;})}</div>}</section>}
        {tab==='methods'&&<div className="method-grid">{keys.map(s=><section key={s} className="panel method-card"><div className="eyebrow">{configs[s].label}</div><h2>{configs[s].name}</h2><div className="score">{models[s]?.validation_score?.toFixed(4)??'—'}<span>{configs[s].metric}</span></div><p>{models[s]?.ready?simpleValidation[s]:'Results loading.'}</p><div className="method-foot"><b>{models[s]?.model?.replaceAll('_',' ')??'Pending'}</b><span>SAMPLE CHECK</span></div></section>)}<div className="method-notice"><ShieldCheck size={20}/><p>Sample scores only. ACV uses 6 labelled cases.</p></div></div>}
        <footer><span><TrainFront size={14}/> Doors Are Closing</span><span>Built for the journeys ahead.</span><span>NEBULA X × LTA · 2026</span></footer>
      </main>
    </div>
  </div>;
}

function ResultView({result:r}:{result:Result}){
  const d=r.details;
  return <div className="result-content">
    <div className="result-summary"><span className="result-icon"><Check size={22}/></span><div><div className="eyebrow">DONE</div><h2>{r.subsystem==='door'?`${d.segments} operations · ${d.abnormal} abnormal`:r.subsystem==='acv'?`Inspect Car ${d.ranking?.[0].car} first`:r.subsystem==='rail'?String(r.rows[0].prediction):`Damage estimate: ${Number(r.rows[0].prediction).toFixed(5)}`}</h2></div><span className="model-tag">{r.model.replaceAll('_',' ')}</span></div>
    <p className="explanation">{d.explanation}</p>
    {d.timeline&&<div className="chart"><h3>Signal timeline</h3><ResponsiveContainer width="100%" height={220}><LineChart data={d.timeline}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="elapsed" type="number" domain={['dataMin','dataMax']}/><YAxis width={48}/><YAxis yAxisId="position" orientation="right" width={48}/><Tooltip/>{d.segment_evidence?.filter(s=>s.label!=='Normal').map((s,i)=><ReferenceArea key={i} x1={s.start} x2={s.end} fill="#d87949" fillOpacity={.2}/>)}<Line name="Motor current (mA)" dataKey="current" stroke="#d87949" dot={false} strokeWidth={2}/><Line yAxisId="position" name="Door position (encoder)" dataKey="position" stroke="#537e85" dot={false} strokeWidth={1.5}/></LineChart></ResponsiveContainer></div>}
    {d.segment_evidence&&<details className="evidence"><summary>Model score <span>Not a risk score</span></summary><div className="table-scroll"><table><thead><tr><th>Start / end (s)</th><th>Class</th><th>Model score</th></tr></thead><tbody>{d.segment_evidence.map((s,i)=><tr key={i}><td>{s.start.toFixed(2)} / {s.end.toFixed(2)}</td><td>{s.label}</td><td>{(s.support*100).toFixed(1)}%</td></tr>)}</tbody></table></div></details>}
    {d.model_support!==undefined&&<p className="explanation">Model score: {(d.model_support*100).toFixed(1)}% — not a risk score.</p>}
    {d.temperature_trend&&<div className="chart"><h3>Cabin temperature</h3><ResponsiveContainer width="100%" height={220}><LineChart data={d.temperature_trend}><XAxis dataKey="sample"/><YAxis domain={['auto','auto']} width={48}/><Tooltip/>{d.ranking?.map((c,i)=><Line key={c.car} name={`Car ${c.car}`} dataKey={c.car} stroke={['#d87949','#537e85','#827090','#728764','#b58a46','#65759b','#a06b76','#778b91'][i]} dot={false}/>)}</LineChart></ResponsiveContainer></div>}
    {d.vibration_preview&&<div className="chart"><h3>Vibration by side</h3><ResponsiveContainer width="100%" height={180}><LineChart data={d.vibration_preview}><XAxis dataKey="time" hide/><YAxis width={48}/><Tooltip/><Line dataKey="side_i" name="Side I" stroke="#d87949" dot={false}/><Line dataKey="side_ii" name="Side II" stroke="#537e85" dot={false}/></LineChart></ResponsiveContainer></div>}
    {d.spectrum&&<div className="chart"><h3>Vibration spectrum</h3><ResponsiveContainer width="100%" height={180}><LineChart data={d.spectrum}><XAxis dataKey="frequency" type="number"/><YAxis width={48}/><Tooltip/><Line dataKey="side_i" name="Side I" stroke="#d87949" dot={false}/><Line dataKey="side_ii" name="Side II" stroke="#537e85" dot={false}/></LineChart></ResponsiveContainer></div>}
    {d.ranking&&<div className="ranking">{d.ranking.map((car,i)=><div key={car.car}><span className="rank">{i+1}</span><b>Car {car.car}</b><div className="rank-bar"><span style={{width:`${Math.max(2,100*car.score/Math.max(1e-6,...d.ranking!.map(c=>c.score)))}%`}}/></div><span>{car.score.toFixed(2)}</span></div>)}</div>}
    {d.heatmap&&<div className="heatmap"><div className="heatmap-title">Axle-box vibration<span>Odd: Side I · Even: Side II</span></div><div className="heatmap-row"><span/><b>01</b><b>02</b><b>03</b><b>04</b><b>05</b><b>06</b><b>07</b><b>08</b></div>{d.heatmap.map((row,i)=><div className="heatmap-row" key={i}><span>CAR {i+1}</span>{row.map((v,j)=><div key={j} title={`Car ${i+1}, position ${j+1}: ${v.toFixed(3)}`} style={{background:`rgba(215,123,76,${.12+.78*v/Math.max(...d.heatmap!.flat())})`}}>{v.toFixed(2)}</div>)}</div>)}</div>}
    {d.stress_preview&&<div className="chart"><h3>Stress history</h3><ResponsiveContainer width="100%" height={180}><LineChart data={d.stress_preview.map((stress,index)=>({stress,index}))}><XAxis dataKey="index" hide/><YAxis width={55}/><Tooltip/><Line dataKey="stress" stroke="#827090" dot={false}/></LineChart></ResponsiveContainer></div>}
    {d.histogram&&<div className="chart"><h3>Stress cycles</h3><ResponsiveContainer width="100%" height={220}><BarChart data={d.histogram.map((n,i)=>({range:d.bin_edges![i].toFixed(1),cycles:n}))}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="range"/><YAxis width={55}/><Tooltip/><Bar dataKey="cycles" fill="#537e85" radius={[3,3,0,0]}/></BarChart></ResponsiveContainer></div>}
    {d.evidence&&<details className="evidence"><summary>Feature evidence</summary><table><thead><tr><th>Signal feature</th><th>Observed</th><th>Reference median</th></tr></thead><tbody>{d.evidence.map(e=><tr key={e.feature}><td>{e.feature.replaceAll('_',' ')}</td><td>{e.value.toPrecision(4)}</td><td>{e.reference.toPrecision(4)}</td></tr>)}</tbody></table></details>}
    <details className="evidence"><summary>Prediction rows <span>{r.rows.length} rows</span></summary><div className="table-scroll"><table><thead><tr>{Object.keys(r.rows[0]).map(k=><th key={k}>{k}</th>)}</tr></thead><tbody>{r.rows.slice(0,100).map((row,i)=><tr key={i}>{Object.values(row).map((v,j)=><td key={j}>{String(v)}</td>)}</tr>)}</tbody></table></div></details>
    {r.warnings.map(w=><div className="result-warning" key={w}><CircleHelp size={15}/>{w}</div>)}
  </div>;
}
