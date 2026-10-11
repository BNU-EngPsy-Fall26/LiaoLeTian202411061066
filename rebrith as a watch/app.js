/* =========================================================
   信号检测论 × PPG 心率监测 交互实验
   3 种干扰情境（环境光 / 佩戴 / 生理）× 15 试次
   每情境：10 信号（强/弱幅度各半随机）+ 5 纯噪声，强弱合并呈现
   ========================================================= */
'use strict';

/* ---------------- 实验配置 ---------------- */
const DUR = 3.2;          // 每试次采样窗口（秒）
const SR  = 100;          // 采样率 Hz
const N   = DUR * SR;     // 320 点
const ANIM_MS = 2600;     // 波形扫描动画时长
const STIM_HOLD_MS = 1500;// 扫描完成后刺激继续可见的固定时长（不可调），到点波形消失

const TPB = 15;           // 每情境 15 试次
const N_SIGNAL = 10;      // 10 个信号试次（强/弱幅度各半随机）
const N_NOISE = 5;        // 5 个噪声试次

/* 难度档位 → 信号幅度（以噪声标准差为单位，≈理论 d′/2）。
   10 个信号试次中强/弱幅度各随机出现一半，但判断与统计合并为「有信号」一类。 */
const DIFF_PRESETS = {
  easy:    { strong:1.80, weak:0.95, label:'简单' },
  normal:  { strong:1.45, weak:0.62, label:'标准' },
  hard:    { strong:1.10, weak:0.45, label:'困难' },
  extreme: { strong:0.85, weak:0.32, label:'极难' }
};
let ampStrong = 1.45, ampWeak = 0.62;   // 运行时被难度选择覆盖

const BLOCKS = [
  {
    short:'环境光干扰',
    name:'情境一 · 环境光干扰',
    badge:'情境 1 / 3',
    kind:'ambient',
    color:'#ffc247',
    scene:'阳光从袖口直射腕部，或头顶荧光灯、屏幕光从表带与皮肤的缝隙漏入光电二极管。'
        + '环境光远强于 LED 反射光，会造成基线缓慢起伏、灯光频率纹波，以及手划过光线时的光突变。',
    tipHTML:'本组 <b>15</b> 个试次：<b style="color:#2dff94">10 个含脉搏信号</b>（脉搏有强有弱，随机出现）、'
          + '<b style="color:#38d6e0">5 个仅噪声</b>，随机排列。请保持<b>中性标准</b>：只在看出周期性脉搏时报「有信号」，不必区分强弱。',
    tipClass:''
  },
  {
    short:'佩戴问题',
    name:'情境二 · 佩戴问题',
    badge:'情境 2 / 3',
    kind:'fit',
    color:'#ff7a5c',
    scene:'表带过松、跑步摆臂或手腕抖动，使传感器与皮肤反复碰撞、瞬时脱离。'
        + '由此产生的运动伪迹是宽带突发脉冲，频谱与真实脉搏高度重叠，还会伴随接触不良的基线跳变与削波——最难与心跳区分。',
    tipHTML:'本组同样是 <b>10 个信号 + 5 个噪声</b>，脉搏强弱随机。运动伪迹可能伪装成脉搏，请仔细辨认<b>周期性</b>，保持中性标准。',
    tipClass:''
  },
  {
    short:'生理因素',
    name:'情境三 · 生理因素',
    badge:'情境 3 / 3',
    kind:'physio',
    color:'#6ea8ff',
    scene:'手腕冰凉、末梢血管收缩使血流灌注指数（PI）下降；同时呼吸会调制每次搏出量，'
        + '自主神经造成心率变异（HRV）——搏动间隔不再严格均匀，脉搏幅度随呼吸强弱起伏，弱搏动甚至接近消失。',
    tipHTML:'本组同样是 <b>10 个信号 + 5 个噪声</b>，脉搏强弱随机。真实脉搏的间隔可能轻微不齐、幅度有起伏，'
          + '不要把周期性不完美的弱脉搏误判为噪声。',
    tipClass:''
  }
];
const STRENGTHS = { signal:'有脉搏信号', none:'仅噪声' };

/* ---------------- 状态 ---------------- */
let bIdx = 0, tIdx = 0;
let blockTrials = [];
let records = [];
let phase = 'idle';        // idle | scanning | ready | feedback
let current = null;
let rafId = null, animStart = 0;
let stimCleared = false, stimTimeout = null;
let lastSummary = null, submitted = false, boardReturnTo = 'screen-welcome', mineUid = null;

const $ = id => document.getElementById(id);

/* ---------------- 数学工具 ---------------- */
function gauss(){
  let u=0,v=0;
  while(u===0) u=Math.random();
  while(v===0) v=Math.random();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}
function clamp(x,a,b){ return x<a?a:x>b?b:x; }
function erf(x){
  const s=x>=0?1:-1; x=Math.abs(x);
  const a1=.254829592,a2=-.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=.3275911;
  const t=1/(1+p*x);
  const y=1-(((((a5*t+a4)*t+a3)*t+a2)*t+a1)*t)*Math.exp(-x*x);
  return s*y;
}
function normCdf(x){ return 0.5*(1+erf(x/Math.SQRT2)); }
// Acklam 逆正态 CDF
function normInv(p){
  const a=[-39.69683028665376,220.9460984245205,-275.9285104469687,138.3577518672690,-30.66479806614716,2.506628277459239];
  const b=[-54.47609879822406,161.5858368580409,-155.6989798598866,66.80131188771972,-13.28068155288572];
  const c=[-.007784894002430293,-.3223964580411365,-2.400758277161838,-2.549732539343734,4.374664141464968,2.938163982698783];
  const d=[.007784695709041462,.3224671290700398,2.445134137142996,3.754408661907416];
  const pl=.02425, ph=1-pl;
  let q,r;
  if(p<pl){ q=Math.sqrt(-2*Math.log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  }
  if(p<=ph){ q=p-.5; r=q*q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
  }
  q=Math.sqrt(-2*Math.log(1-p));
  return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
}
function shuffle(a){ for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]];} return a; }

/* ---------------- PPG 信号合成 ---------------- */
// 单周期脉搏波形状：收缩期主峰 + 重搏波（参数为周期内相位 0–1）
function pulseShape(ph){
  return 1.0*Math.exp(-Math.pow((ph-0.15)/0.060,2))
       + 0.34*Math.exp(-Math.pow((ph-0.40)/0.095,2));
}
/* 生成逐搏时间序列（支持 HRV），返回搏动时刻数组 */
function beatTimes(kind){
  const baseHR=58+Math.random()*38;   // 58–96 BPM
  let hr=baseHR, t=-Math.random()*(60/baseHR);
  const beats=[];
  const jit = kind==='physio'?0.085:0.02;   // 逐搏间隔抖动
  while(t<DUR){
    t += (60/hr)*(1+jit*gauss());
    beats.push(t);
    if(kind==='physio') hr=clamp(hr+0.9*gauss(), 50, 105);  // 心率缓慢游走
  }
  return {beats, hr0:Math.round(baseHR)};
}

/* ---------------- 三种情境噪声 ---------------- */
function synthNoise(kind){
  const white=Array.from({length:N},gauss);
  const col=white.map((w,i)=>((white[i-1]??w)+2*w+(white[i+1]??w))/4); // 轻度低通
  const out=new Float64Array(N);

  if(kind==='ambient'){
    const f1=.10+Math.random()*.22, f2=.30+Math.random()*.25;
    const p1=Math.random()*6.28, p2=Math.random()*6.28;
    const rippleF=5.5+Math.random()*2.5, rippleP=Math.random()*6.28;
    const doFlash=Math.random()<.75;
    const tc=.5+Math.random()*2.1, sgn=Math.random()<.5?-1:1, fAmp=1.0+Math.random()*.6;
    for(let i=0;i<N;i++){
      const t=i/SR;
      let v=col[i]
        + .60*Math.sin(2*Math.PI*f1*t+p1)
        + .34*Math.sin(2*Math.PI*f2*t+p2)
        + .18*Math.sin(2*Math.PI*rippleF*t+rippleP);
      if(doFlash && t>tc) v += sgn*fAmp*Math.exp(-(t-tc)/0.55);  // 光突变后 AGC 缓慢恢复
      out[i]=v;
    }
  } else if(kind==='fit'){
    const nM=1+(Math.random()<.6?1:0);
    const motions=[];
    for(let k=0;k<nM;k++) motions.push({
      c:.5+Math.random()*2.1, w:.10+Math.random()*.16,
      a:2.1+Math.random()*1.0, f:8+Math.random()*7, s:Math.random()<.5?-1:1
    });
    const doStep=Math.random()<.7;
    const sc=.4+Math.random()*1.6, sAmp=(.6+Math.random()*.5)*(Math.random()<.5?-1:1);
    const sw=.5+Math.random()*.8;
    for(let i=0;i<N;i++){
      const t=i/SR;
      let v=1.15*col[i];
      for(const m of motions)
        v += m.s*m.a*Math.sin(2*Math.PI*m.f*t)*Math.exp(-Math.pow((t-m.c)/m.w,2));
      if(doStep && t>sc && t<sc+sw) v += sAmp;    // 接触压力变化→基线跳变
      v=clamp(v,-2.7,2.7);                        // 瞬态削波
      out[i]=v;
    }
  } else { // physio：噪声温和，偶发小幅肌电
    const f=.26+Math.random()*0.06, ph=Math.random()*6.28;
    const doEmg=Math.random()<.6;
    const c=.7+Math.random()*1.7, w=.08+Math.random()*.1, a=.7+Math.random()*.5;
    for(let i=0;i<N;i++){
      const t=i/SR;
      let v=.9*col[i] + .22*Math.sin(2*Math.PI*f*t+ph);
      if(doEmg) v += a*Math.sin(2*Math.PI*13*t)*Math.exp(-Math.pow((t-c)/w,2));
      out[i]=v;
    }
  }
  // 标定噪声标准差为 1，使「幅度 = 信噪比」语义跨情境一致
  let s=0; for(let i=0;i<N;i++) s+=out[i]*out[i];
  const sd=Math.sqrt(s/N);
  for(let i=0;i<N;i++) out[i]/=sd;
  return out;
}

function makeTrial(kind, strength){
  const hasSignal=strength==='signal';
  // 信号试次的脉搏幅度在强/弱两档间随机（各约一半）
  const amp=hasSignal ? (Math.random()<0.5?ampStrong:ampWeak) : 0;
  const {beats,hr0}=hasSignal?beatTimes(kind):{beats:null,hr0:null};
  let sig=new Float64Array(N);
  if(hasSignal){
    const fResp=0.26+Math.random()*0.08, phResp=Math.random()*6.28;
    const mResp=kind==='physio'?0.45:0.10;
    for(let i=0;i<N;i++){
      const t=i/SR;
      let v=0;
      for(const tb of beats){
        const p=60/hr0, ph=(t-tb)/p;
        if(ph>0&&ph<1){
          const ba=amp*Math.max(0.15,1+mResp*Math.sin(2*Math.PI*fResp*tb+phResp))*(1+0.06*gauss());
          v+=ba*pulseShape(ph);
        }
      }
      sig[i]=v;
    }
    if(kind==='fit'&&Math.random()<.75){
      const a=.3+Math.random()*1.8, b=a+.25+Math.random()*.55;
      for(let i=0;i<N;i++){ const t=i/SR; if(t>a&&t<b) sig[i]*=.12; }
    }
    let mean=0; for(let i=0;i<N;i++) mean+=sig[i]; mean/=N;
    for(let i=0;i<N;i++) sig[i]-=mean;
  }
  const noise=synthNoise(kind);
  const mix=new Float64Array(N);
  for(let i=0;i<N;i++) mix[i]=sig[i]+noise[i];
  return {kind,strength,hasSignal,hr:hr0,sig,noise,mix};
}

/* ---------------- 画布基础 ---------------- */
function hiDPI(cv){
  const dpr=window.devicePixelRatio||1;
  const w=cv.width, h=cv.height;
  cv.width=Math.round(w*dpr); cv.height=Math.round(h*dpr);
  cv.style.aspectRatio=`${w} / ${h}`;
  const ctx=cv.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx,w,h};
}
const TRACE_M={l:40,r:14,t:18,b:26};
function traceXY(w,h){
  const pw=w-TRACE_M.l-TRACE_M.r, ph=h-TRACE_M.t-TRACE_M.b;
  return {
    x:i=>TRACE_M.l + (i/(N-1))*pw,
    y:v=>TRACE_M.t + ph/2 - v*(ph/10)
  };
}
function drawGrid(ctx,w,h){
  const {x,y}=traceXY(w,h);
  ctx.save();
  ctx.lineWidth=1;
  ctx.font='10px Consolas,monospace';
  ctx.fillStyle='#3d6b5b';
  ctx.strokeStyle='rgba(45,120,90,.18)';
  for(let u=-4;u<=4;u++){
    ctx.beginPath(); ctx.moveTo(TRACE_M.l,y(u)); ctx.lineTo(w-TRACE_M.r,y(u)); ctx.stroke();
    ctx.fillText(String(u),10,y(u)+3);
  }
  ctx.strokeStyle='rgba(45,255,148,.25)';
  ctx.beginPath(); ctx.moveTo(TRACE_M.l,y(0)); ctx.lineTo(w-TRACE_M.r,y(0)); ctx.stroke();
  for(let s=0;s<=DUR+1e-6;s+=0.4){
    const i=s*SR;
    ctx.strokeStyle='rgba(45,120,90,.18)';
    ctx.beginPath(); ctx.moveTo(x(i),TRACE_M.t); ctx.lineTo(x(i),h-TRACE_M.b); ctx.stroke();
    ctx.fillStyle='#3d6b5b';
    ctx.fillText(s.toFixed(1)+'s',x(i)-11,h-10);
  }
  ctx.restore();
}
function drawTrace(cv, mix, sig, upto, showOverlay){
  const {ctx,w,h}=cv._view;
  ctx.clearRect(0,0,w,h);
  drawGrid(ctx,w,h);
  const {x,y}=traceXY(w,h);
  const idx=Math.max(1,Math.floor(upto*(N-1)));
  if(showOverlay && sig){
    ctx.save();
    ctx.strokeStyle='rgba(255,194,71,.85)';
    ctx.setLineDash([5,4]); ctx.lineWidth=1.4;
    ctx.beginPath();
    for(let i=0;i<=idx;i++){ const px=x(i),py=y(sig[i]); i?ctx.lineTo(px,py):ctx.moveTo(px,py); }
    ctx.stroke();
    ctx.restore();
  }
  ctx.save();
  ctx.strokeStyle='#2dff94';
  ctx.lineWidth=2;
  ctx.shadowColor='rgba(45,255,148,.55)';
  ctx.shadowBlur=7;
  ctx.beginPath();
  for(let i=0;i<=idx;i++){ const px=x(i),py=y(mix[i]); i?ctx.lineTo(px,py):ctx.moveTo(px,py); }
  ctx.stroke();
  ctx.restore();
  if(upto<1){
    const lx=x(idx);
    ctx.save();
    ctx.strokeStyle='rgba(45,255,148,.18)';
    ctx.beginPath(); ctx.moveTo(lx,TRACE_M.t); ctx.lineTo(lx,h-TRACE_M.b); ctx.stroke();
    ctx.fillStyle='#aaffd2';
    ctx.shadowColor='#2dff94'; ctx.shadowBlur=12;
    ctx.beginPath(); ctx.arc(lx,y(mix[idx]),3.6,0,6.283); ctx.fill();
    ctx.restore();
  }
}

/* ---------------- 屏幕切换 ---------------- */
function show(id){
  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
  $(id).classList.add('active');
  window.scrollTo({top:0});
}

/* ---------------- 欢迎页示例 ---------------- */
function drawDemos(){
  $('demo-signal')._view=hiDPI($('demo-signal'));
  $('demo-noise')._view=hiDPI($('demo-noise'));
  const tS=makeTrial('ambient','signal'), tN=makeTrial('ambient','none');
  drawTrace($('demo-signal'),tS.mix,null,1,false);
  drawTrace($('demo-noise'), tN.mix,null,1,false);
}

/* ---------------- 实验流程 ---------------- */
function startExperiment(){
  const d=DIFF_PRESETS[$('sel-diff').value];
  ampStrong=d.strong; ampWeak=d.weak;
  bIdx=0;tIdx=0;records=[];
  showBlockIntro();
}
function buildBlockTrials(){
  const kind=BLOCKS[bIdx].kind;
  const flags=[];
  for(let i=0;i<N_SIGNAL;i++) flags.push('signal');
  for(let i=0;i<N_NOISE;i++) flags.push('none');
  shuffle(flags);
  blockTrials=flags.map(s=>makeTrial(kind,s));
}
function showBlockIntro(){
  const b=BLOCKS[bIdx];
  $('bi-badge').textContent=b.badge;
  $('bi-title').textContent=b.name;
  $('bi-scene').textContent=b.scene;
  const tip=$('bi-tip');
  tip.className='condition-tip '+b.tipClass;
  tip.innerHTML=b.tipHTML;
  show('screen-blockintro');
}
function beginBlock(){
  buildBlockTrials();
  tIdx=0;
  show('screen-trial');
  nextTrial();
}
function nextTrial(){
  current=blockTrials[tIdx];
  phase='scanning';
  $('feedback').classList.add('hidden');
  stimCleared=false;
  $('btn-noise').disabled=true;
  $('btn-signal').disabled=true;
  $('t-badge').textContent=BLOCKS[bIdx].badge;
  $('t-blockname').textContent=BLOCKS[bIdx].name;
  $('t-counter').textContent=`试次 ${tIdx+1} / ${TPB}`;
  $('progress-fill').style.width=(tIdx/TPB*100)+'%';
  const ind=$('scan-indicator'); ind.className='led-indicator'; ind.innerHTML='<i></i>采样中';
  const ms=$('monitor-status'); ms.className='monitor-status'; ms.textContent='等待波形扫描…（2.6 s）';
  if(!$('trace-canvas')._view) $('trace-canvas')._view=hiDPI($('trace-canvas'));
  animStart=performance.now();
  cancelAnimationFrame(rafId);
  const loop=function(now){
    const upto=Math.min(1,(now-animStart)/ANIM_MS);
    drawTrace($('trace-canvas'),current.mix,null,upto,false);
    if(upto<1){ rafId=requestAnimationFrame(loop); }
    else enableResponse();
  };
  rafId=requestAnimationFrame(loop);
}
function enableResponse(){
  phase='ready';
  stimCleared=false;
  $('btn-noise').disabled=false;
  $('btn-signal').disabled=false;
  const ind=$('scan-indicator'); ind.className='led-indicator ready'; ind.innerHTML='<i></i>请判断';
  const ms=$('monitor-status'); ms.className='monitor-status ready';
  ms.textContent=`刺激呈现中（${(STIM_HOLD_MS/1000).toFixed(1)} s 后消失）—— 是否含有脉搏信号？（F / J）`;
  // 脉搏呈现限时：到点波形消失（不可调）
  clearTimeout(stimTimeout);
  stimTimeout=setTimeout(clearStim, STIM_HOLD_MS);
}
function clearStim(){
  if(stimCleared) return;
  stimCleared=true;
  const cv=$('trace-canvas'), v=cv._view;
  if(v){ v.ctx.clearRect(0,0,v.w,v.h);
    v.ctx.save();
    v.ctx.fillStyle='#5d8a78'; v.ctx.font='14px "Segoe UI",sans-serif';
    v.ctx.textAlign='center'; v.ctx.textBaseline='middle';
    v.ctx.fillText('刺激已消失 —— 请根据记忆判断', v.w/2, v.h/2);
    v.ctx.restore();
  }
  const ms=$('monitor-status'); ms.className='monitor-status'; ms.textContent='刺激已消失 —— 请根据记忆判断（F / J）';
}
function clearStimTimer(){ clearTimeout(stimTimeout); stimTimeout=null; }
function respond(saySignal){
  if(phase!=='ready') return;
  clearStimTimer();
  phase='feedback';
  const truth=current.hasSignal;
  const outcome = truth&&saySignal ? 'hit' : truth&&!saySignal ? 'miss'
               : !truth&&saySignal ? 'fa' : 'cr';
  records.push({
    block:bIdx, kind:current.kind, strength:current.strength, trial:tIdx,
    signal:truth, response:saySignal, outcome, hr:current.hr
  });
  $('btn-noise').disabled=true; $('btn-signal').disabled=true;
  $('btn-noise').blur(); $('btn-signal').blur();
  const ind=$('scan-indicator'); ind.className='led-indicator done'; ind.innerHTML='<i></i>已记录';
  $('monitor-status').className='monitor-status';
  $('monitor-status').textContent=truth
    ?'答案揭晓 —— 橙色虚线为该试次中隐藏的纯脉搏成分'
    :'答案揭晓 —— 该试次不含脉搏，所有起伏均为干扰噪声';
  drawTrace($('trace-canvas'),current.mix,truth?current.sig:null,1,truth);

  const map={
    hit:['chip-hit','击中 HIT','这段波形确实含有规律的心跳脉搏，你正确发现了它！'],
    miss:['chip-miss','漏报 MISS','这段波形其实藏有脉搏（约 '+current.hr+' BPM），但它被你忽略了。'],
    fa:['chip-fa','虚报 FALSE ALARM','这段只是干扰噪声，你被其中的随机起伏欺骗了。'],
    cr:['chip-cr','正确拒斥 C.R.','这段只有干扰噪声，你正确地排除了它。']
  };
  const chip=$('fb-chip'); chip.className='outcome-chip '+map[outcome][0]; chip.textContent=map[outcome][1];
  $('fb-truth').innerHTML=`实际：<b>${STRENGTHS[current.strength]}</b>　·　${map[outcome][2]}`;
  $('feedback').classList.remove('hidden');
  $('progress-fill').style.width=((tIdx+1)/TPB*100)+'%';
}
function advance(){
  if(phase!=='feedback') return;
  if(tIdx+1<TPB){ tIdx++; nextTrial(); }
  else if(bIdx+1<BLOCKS.length){ bIdx++; showBlockIntro(); }
  else{ finishExperiment(); }
}

/* ---------------- 结果统计 ---------------- */
// 每情境：10 个信号试次（强弱幅度混合）+ 5 个噪声试次，合并计算一组指标
function cellStats(list){
  const sig=list.filter(r=>r.signal), noi=list.filter(r=>!r.signal);
  const ns=sig.length, nn=noi.length;
  const hits=sig.filter(r=>r.response).length;
  const fa=noi.filter(r=>r.response).length;
  const miss=ns-hits, cr=nn-fa;
  const Hz=(hits===0||hits===ns)?(hits+.5)/(ns+1):hits/ns;
  const Fz=(fa===0||fa===nn)?(fa+.5)/(nn+1):fa/nn;
  const zh=normInv(Hz), zf=normInv(Fz);
  return {
    ns,nn,hits,miss,fa,cr,
    H:hits/ns,F:fa/nn,
    d:zh-zf, c:-(zh+zf)/2, beta:Math.exp((zf*zf-zh*zh)/2)
  };
}
const pct=v=>(v*100).toFixed(0)+'%';
const f2=v=>v.toFixed(2);

function finishExperiment(){
  show('screen-results');
  // rows：每情境一行，共 3 行
  const rows=BLOCKS.map((_,bi)=>({bi, s:cellStats(records.filter(r=>r.block===bi))}));

  const totAcc=records.filter(r=>r.outcome==='hit'||r.outcome==='cr').length/records.length;
  const sigAll=records.filter(r=>r.signal), noiAll=records.filter(r=>!r.signal);
  const diffLabel=DIFF_PRESETS[$('sel-diff').value].label;
  const avgD=rows.reduce((a,r)=>a+r.s.d,0)/3;
  $('results-overall').innerHTML=
    `难度：<b>${diffLabel}</b>（信号幅度在 ${ampWeak.toFixed(2)}–${ampStrong.toFixed(2)} 间随机混合），脉搏呈现 <b>1.5 s</b> 后消失。`+
    `共完成 ${records.length} 个试次（3 情境 × 15，每情境 10 信号 + 5 噪声）。`+
    `整体正确率 <b>${pct(totAcc)}</b>，击中率 <b>${pct(sigAll.filter(r=>r.response).length/sigAll.length)}</b>，`+
    `虚报率 <b>${pct(noiAll.filter(r=>r.response).length/noiAll.length)}</b>，三情境平均 d′≈<b>${f2(avgD)}</b>。`+
    `出现 0% 或 100% 时 d′ 已做 log-linear 校正（+0.5/+1）。`;

  // 记录本次成绩摘要供排行榜提交
  lastSummary={
    diffLabel, avgD, acc:totAcc,
    H:sigAll.filter(r=>r.response).length/sigAll.length,
    F:noiAll.filter(r=>r.response).length/noiAll.length
  };
  submitted=false;
  const sb=$('btn-submit'); sb.disabled=false; sb.textContent='🏆 提交成绩到排行榜';

  const tb=$('results-table').querySelector('tbody');
  tb.innerHTML='';
  rows.forEach(r=>{
    const tr=document.createElement('tr');
    tr.innerHTML=
      `<td><span style="color:${BLOCKS[r.bi].color}">${BLOCKS[r.bi].short}</span></td>`+
      `<td style="color:var(--green)">${r.s.hits}</td><td style="color:var(--amber)">${r.s.miss}</td>`+
      `<td style="color:var(--red)">${r.s.fa}</td><td style="color:var(--teal)">${r.s.cr}</td>`+
      `<td class="mono">${pct(r.s.H)}</td><td class="mono">${pct(r.s.F)}</td>`+
      `<td class="mono"><b>${f2(r.s.d)}</b></td><td class="mono">${f2(r.s.c)}</td>`+
      `<td class="mono">${f2(r.s.beta)}</td>`;
    tb.appendChild(tr);
  });

  drawROC(rows);
  drawBars(rows);
  writeInterpretation(rows, {avgD, totAcc,
    H:sigAll.filter(r=>r.response).length/sigAll.length,
    F:noiAll.filter(r=>r.response).length/noiAll.length});
}

/* ---------------- ROC 图 ---------------- */
function drawROC(rows){
  const cv=hiDPI($('roc-canvas'));
  const {ctx,w,h}=cv;
  ctx.clearRect(0,0,w,h);
  const M={l:48,r:18,t:54,b:44}, pw=w-M.l-M.r, ph=h-M.t-M.b;
  const X=v=>M.l+v*pw, Y=v=>M.t+(1-v)*ph;
  ctx.strokeStyle='rgba(120,180,150,.15)'; ctx.lineWidth=1;
  ctx.font='10px Consolas,monospace';
  for(let v=0;v<=1.0001;v+=.2){
    ctx.beginPath();ctx.moveTo(X(v),M.t);ctx.lineTo(X(v),M.t+ph);ctx.stroke();
    ctx.beginPath();ctx.moveTo(M.l,Y(v));ctx.lineTo(M.l+pw,Y(v));ctx.stroke();
    ctx.fillStyle='#7fa393';
    ctx.fillText(v.toFixed(1),X(v)-9,h-26);
    ctx.fillText(v.toFixed(1),14,Y(v)+3);
  }
  ctx.strokeStyle='rgba(200,220,210,.35)'; ctx.setLineDash([5,4]);
  ctx.beginPath();ctx.moveTo(X(0),Y(0));ctx.lineTo(X(1),Y(1));ctx.stroke();
  ctx.setLineDash([]);
  // 等 d′ 参考曲线：三情境平均 d′
  const avgD=rows.reduce((a,r)=>a+r.s.d,0)/3;
  ctx.strokeStyle='rgba(45,255,148,.6)';ctx.lineWidth=1.6;ctx.beginPath();
  for(let i=0;i<=100;i++){
    const x=.02+i/100*(.98-.02);
    const y=normCdf(avgD+normInv(x));
    i?ctx.lineTo(X(x),Y(y)):ctx.moveTo(X(x),Y(y));
  }
  ctx.stroke();
  // 实测点：形状=情境，颜色=情境色
  const marker=(bi)=>bi===0?'circle':bi===1?'square':'triangle';
  rows.forEach(r=>{
    const px=X(r.s.F), py=Y(r.s.H);
    const color=BLOCKS[r.bi].color;
    ctx.fillStyle=color; ctx.strokeStyle='#06110c'; ctx.lineWidth=1.5;
    ctx.shadowColor=color; ctx.shadowBlur=7;
    ctx.beginPath();
    if(marker(r.bi)==='circle') ctx.arc(px,py,6.5,0,6.283);
    else if(marker(r.bi)==='square') ctx.rect(px-5.5,py-5.5,11,11);
    else{ ctx.moveTo(px,py-7.5);ctx.lineTo(px+7,py+5.5);ctx.lineTo(px-7,py+5.5);ctx.closePath(); }
    ctx.fill(); ctx.shadowBlur=0; ctx.stroke();
  });
  // 图例
  ctx.font='11px sans-serif'; ctx.textAlign='left'; ctx.textBaseline='middle';
  const ly=M.t-30;
  const leg=(x,shape,color,text)=>{
    ctx.fillStyle=color;
    ctx.beginPath();
    if(shape==='circle') ctx.arc(x+5,ly,5,0,6.283);
    else if(shape==='square') ctx.rect(x,ly-5,10,10);
    else { ctx.moveTo(x+5,ly-6);ctx.lineTo(x+11,ly+5);ctx.lineTo(x-1,ly+5);ctx.closePath(); }
    ctx.fill();
    ctx.fillStyle='#cfe6da';
    ctx.fillText(text,x+16,ly+1);
  };
  leg(M.l,'circle',BLOCKS[0].color,'● 环境光');
  leg(M.l+86,'square',BLOCKS[1].color,'■ 佩戴问题');
  leg(M.l+172,'triangle',BLOCKS[2].color,'▲ 生理因素');
  ctx.strokeStyle='rgba(45,255,148,.6)'; ctx.setLineDash([5,4]);
  ctx.beginPath();ctx.moveTo(M.l,M.t-46);ctx.lineTo(M.l+22,M.t-46);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#7fa393';ctx.fillText('等 d′ 曲线（三情境平均 d′='+f2(avgD)+'）',M.l+28,M.t-45);
  // 轴标题
  ctx.fillStyle='#9fc4b4';ctx.font='12px sans-serif';ctx.textAlign='center';ctx.textBaseline='alphabetic';
  ctx.fillText('虚报率 P(FA)',M.l+pw/2,h-6);
  ctx.save();ctx.translate(14,M.t+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText('命中率 P(H)',0,0);ctx.restore();
  ctx.textAlign='left';
  $('roc-cap').innerHTML=
    '每个情境一个点（P(FA), P(H)）。点越靠左上（命中率高且虚报率低），d′ 越大；三点在等 d′ 曲线附近的上下位置差异反映三种干扰的可分辨程度。';
}

/* ---------------- d′ 柱状图 ---------------- */
function drawBars(rows){
  const cv=hiDPI($('bar-canvas'));
  const {ctx,w,h}=cv;
  ctx.clearRect(0,0,w,h);
  const M={l:46,r:14,t:30,b:50}, pw=w-M.l-M.r, ph=h-M.t-M.b;
  const yMin=0,yMax=3.5;
  const Y=v=>M.t+(yMax-v)/(yMax-yMin)*ph;
  ctx.font='10px Consolas,monospace';ctx.lineWidth=1;
  for(let v=0;v<=3;v+=.5){
    ctx.strokeStyle= v===0?'rgba(200,220,210,.5)':'rgba(120,180,150,.15)';
    ctx.beginPath();ctx.moveTo(M.l,Y(v));ctx.lineTo(M.l+pw,Y(v));ctx.stroke();
    ctx.fillStyle='#7fa393';ctx.fillText(v.toFixed(1),14,Y(v)+3);
  }
  const groupW=pw/3, barW=46;
  rows.forEach(r=>{
    const cx=M.l+groupW*(r.bi+.5);
    const x=cx-barW/2, yv=Y(Math.max(0,r.s.d));
    ctx.fillStyle=BLOCKS[r.bi].color;
    ctx.fillRect(x,yv,barW,Y(0)-yv);
    ctx.fillStyle='#eafcf4';ctx.font='bold 13px Consolas,monospace';ctx.textAlign='center';
    ctx.fillText(f2(r.s.d),cx,yv-6);
    ctx.fillStyle=BLOCKS[r.bi].color;ctx.font='bold 12px sans-serif';
    ctx.fillText(BLOCKS[r.bi].short,cx,h-26);
    ctx.textAlign='left';
  });
  $('bar-cap').innerHTML='各情境合并强/弱信号后的整体 d′。情境间高低差异反映不同噪声结构与脉搏频谱的重叠程度（运动伪迹通常最难分辨）。';
}

/* ---------------- 风趣评价 ---------------- */
function funnyEval({avgD,acc,H,F}){
  // 段位 + 头衔
  const tiers=[
    {d:2.8, title:'🧠 PPG 总设计师', color:'#ffd54a',
     desc:'你对噪声的嗅觉比光电二极管还敏锐。小米、华为、佳明 HR 部门的 Offer 已经在路上了——建议提薪谈判时把这份成绩截屏。'},
    {d:2.0, title:'🔬 临床级读图专家', color:'#2dff94',
     desc:'运动伪迹在你面前像裸奔。你闭着眼睛都能区分「心跳」和「甩手」，真实手环的算法如果会上网冲浪，大概会默默给你点了个赞。'},
    {d:1.2, title:'💪 资深智能手表质检员', color:'#6ea8ff',
     desc:'你有一颗合格的检测心。虽然偶尔会把腕部悬垂误判成心动过速，但瑕不掩瑜，你的表现已经甩开了随机猜测一大截。'},
    {d:0.5, title:'😵 信号盲区探险家', color:'#ffc247',
     desc:'你对波形的判断比天气预报还飘忽。好消息是：你可能不适合读图，坏消息是——你正在读的图还就是天气。建议多吃蔬菜、早睡早起，争取下次把 d′ 拉到正数。'},
    {d:-9,  title:'🎲 量子随机数生成器', color:'#ff7a5c',
     desc:'你的反应严格遵循薛定谔分布：既击中了又漏报了。建议下次直接掷骰子，说不定还能省点脑力。不过别担心，至少你提供了极有价值的对照组数据。'}
  ];
  const tier=tiers.find(t=>avgD>=t.d);
  // 虚报吐槽
  let faRoast='';
  if(F>.55) faRoast='虚报率高到像在抢红包——看见信号就按。真实手环若采用你这标准，每天至少弹 40 次心率异常警告，电池三小时告急。';
  else if(F>.35) faRoast='虚报率略高，给人一种「宁可错杀三千，不可放过一个心跳」的豪迈。建议改行做安检门，会更适合这份宁多勿少的情怀。';
  else if(F<.05) faRoast='虚报率极低，堪称「宁可沉默也不说谎」的保守派。如果手表是你的下属，它大概会因为太怕报错而干脆什么都不报——省电模式本模式。';

  // 击中率吐槽
  let hitRoast='';
  if(H>.90) hitRoast='击中率惊人，连最微弱的脉搏都逃不过你的法眼。建议去海关 X 光机岗，国家需要你这样的人才。';
  else if(H<.45) hitRoast='击中率有点惨淡，脉搏在你面前玩起了隐身术。好消息：你极不容易虚报；坏消息：你也不太容易发现真相。';

  // 情境补刀
  const diffLabel=DIFF_PRESETS[$('sel-diff').value].label;
  let diffNote='';
  if(diffLabel==='极难' && avgD<0.8) diffNote='顺带一提，极难档的脉搏确实只有蚊子心跳那么大，没全灭已经算及格了。';
  else if(diffLabel==='简单' && avgD<1.5) diffNote='顺带一提，你选的是「简单」模式。理论上这应该是「闭着眼睛都能过」的段位，而你……好像没闭眼，那问题出在哪了呢？';

  // 最终结语（高分 / 低分二选一）
  let conclusion;
  if(avgD>=1.5){
    conclusion={text:'你已合格，可以开启新生活了。',tone:'pass'};
  } else {
    conclusion={text:'恭喜你，被报废了，可以继续重开了。',tone:'fail'};
  }

  return {tier,faRoast,hitRoast,diffNote,conclusion};
}

/* ---------------- 文字解读 ---------------- */
function writeInterpretation(rows, overall){
  const d=bi=>rows.find(r=>r.bi===bi).s.d;
  const env=d(0), fit=d(1), phy=d(2);
  const order=[['环境光干扰',env],['佩戴问题',fit],['生理因素',phy]]
    .sort((a,b)=>b[1]-a[1]);
  const {tier,faRoast,hitRoast,diffNote,conclusion}=funnyEval(overall);
  const conclColor=conclusion.tone==='pass'?'var(--green)':'var(--red)';
  $('interpret').innerHTML=
    '<div id="eval-card" class="eval-card" style="margin:24px 0;padding:18px 20px;background:rgba(45,255,148,.04);border:1px solid rgba(45,255,148,.2);border-radius:12px">'+
    '<div style="font-size:1.3rem;font-weight:800;margin-bottom:6px;color:'+tier.color+'">'+tier.title+'</div>'+
    '<div style="font-size:.95rem;line-height:1.7;color:var(--text-soft);margin-bottom:12px">'+tier.desc+'</div>'+
    (faRoast?'<div style="font-size:.9rem;line-height:1.6;color:var(--muted);margin-bottom:10px"><span style="color:var(--red)">虚报点评：</span>'+faRoast+'</div>':'')+
    (hitRoast?'<div style="font-size:.9rem;line-height:1.6;color:var(--muted);margin-bottom:10px"><span style="color:var(--green)">击中点评：</span>'+hitRoast+'</div>':'')+
    (diffNote?'<div style="font-size:.9rem;line-height:1.6;color:var(--muted);margin-bottom:14px"><span style="color:var(--amber)">难度彩蛋：</span>'+diffNote+'</div>':'')+
    '<div style="font-size:1.15rem;font-weight:700;padding:12px 14px;border-radius:8px;background:'+(conclusion.tone==='pass'?'rgba(45,255,148,.08)':'rgba(255,122,92,.08)')+';color:'+conclColor+';text-align:center;margin-bottom:14px">'+conclusion.text+'</div>'+
    '<div style="text-align:center"><button id="btn-eval-ok" class="btn btn-primary">我知道了</button></div>'+
    '</div>'+
    '<div id="eval-rest" style="display:none">'+
    '<h4>结果解读</h4>'+
    '<p><b>干扰情境：</b>你的辨别力排序为 '+
      order.map((o,i)=>`${i+1}. ${o[0]}（d′=${f2(o[1])}）`).join(' ＞ ')+
      '。信号试次的脉搏幅度在强、弱两档间随机混合，因此 d′ 反映的是你在该情境下对脉搏信号的<b>整体感受性</b>。'+
      '这一排序对应三种干扰的物理机制：'+
      '<b style="color:#ffc247">环境光</b>主要是低频漂移与偶发光突变，频率远离脉搏（约 1–1.6 Hz），较容易凭周期性识破；'+
      '<b style="color:#ff7a5c">佩戴问题</b>产生的运动伪迹是与脉搏同频段的宽带突发脉冲，最容易伪装成心跳，通常最难分辨；'+
      '<b style="color:#6ea8ff">生理因素</b>下真实搏动本身幅度起伏、间隔轻微不齐（HRV 与呼吸调制），削弱了「规律」这一线索。</p>'+
    '<p class="ppg-tie"><b>对应到真实手环：</b>环境光干扰靠光学屏蔽、高占空比 LED 与自适应滤波（AGC）抑制；'+
      '运动伪迹需借助加速度计做运动相消（如手环的运动补偿算法），是心率检测的最大难题；'+
      '生理低灌注则要求更高灵敏度的模拟前端与可调光强。信号检测论告诉我们：这些手段本质上都在努力提高 d′，'+
      '而提醒模式 / 省电模式的选择，则是在给定 d′ 下沿同一条 ROC 曲线移动判断标准 β。</p>'+
    '</div>';
  $('btn-eval-ok').addEventListener('click',()=>{
    const card=$('eval-card'); card.style.transition='opacity .3s, transform .3s';
    card.style.opacity='0'; card.style.transform='translateY(-8px)';
    setTimeout(()=>{
      card.style.display='none';
      const rest=$('eval-rest'); if(rest){ rest.style.display='block'; rest.style.opacity='0'; rest.style.transition='opacity .35s'; requestAnimationFrame(()=>rest.style.opacity='1'); }
    },300);
  });
}

/* ---------------- CSV 导出 ---------------- */
function exportCSV(){
  const head=['situation_no','situation','trial_in_situation','signal_present','response_signal','outcome','hr_bpm'];
  const rows=records.map(r=>[
    r.block+1, BLOCKS[r.block].short, r.trial+1,
    r.signal?1:0, r.response?1:0,
    {hit:'击中',miss:'漏报',fa:'虚报',cr:'正确拒斥'}[r.outcome],
    r.hr??''
  ]);
  const csv=[head,...rows].map(row=>row.map(v=>{
    const s=String(v??'');
    return /[,"\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;
  }).join(',')).join('\r\n');
  const blob=new Blob(['\uFEFF'+csv],{type:'text/csv;charset=utf-8'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='ppg_sdt_3situations.csv';
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ---------------- 排行榜（localStorage） ---------------- */
const LB_KEY='ppg_sdt_leaderboard_v1', ID_KEY='ppg_sdt_player_id';

function escHtml(s){
  return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function loadBoard(){ try{ const a=JSON.parse(localStorage.getItem(LB_KEY)); return Array.isArray(a)?a:[]; }catch(e){ return []; } }
function saveBoard(a){ localStorage.setItem(LB_KEY, JSON.stringify(a.slice(0,100))); }
function sortedBoard(a){ return a.slice().sort((x,y)=> (y.avgD-x.avgD) || (y.acc-x.acc)); }
function fmtTime(ts){
  const d=new Date(ts), p=n=>String(n).padStart(2,'0');
  return `${d.getMonth()+1}/${d.getDate()} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
function renderBoard(){
  const b=sortedBoard(loadBoard());
  const tb=$('board-table').querySelector('tbody');
  tb.innerHTML='';
  $('board-empty').style.display=b.length?'none':'block';
  b.forEach((e,i)=>{
    const tr=document.createElement('tr');
    if(i<3) tr.classList.add('rank-top');
    if(e.uid===mineUid) tr.classList.add('board-mine');
    const medal=i===0?'🥇':i===1?'🥈':i===2?'🥉':'';
    tr.innerHTML=
      `<td class="rank-cell rank-${i+1}">${medal||'#'+(i+1)}</td>`+
      `<td><b>${escHtml(e.id)}</b></td>`+
      `<td>${escHtml(e.diff)}</td>`+
      `<td class="mono"><b>${f2(e.avgD)}</b></td>`+
      `<td class="mono">${pct(e.acc)}</td>`+
      `<td class="mono">${pct(e.H)}</td>`+
      `<td class="mono">${pct(e.F)}</td>`+
      `<td class="mono">${fmtTime(e.date)}</td>`;
    tb.appendChild(tr);
  });
}
function openBoard(ret){ boardReturnTo=ret; renderBoard(); show('screen-board'); }
function submitScore(){
  if(submitted||!lastSummary) return;
  let id=($('inp-id').value||'').trim().replace(/[<>]/g,'').slice(0,12)||'匿名选手';
  $('inp-id').value=id;
  try{ localStorage.setItem(ID_KEY,id); }catch(e){}
  const uid=Date.now()+'-'+Math.random().toString(36).slice(2,8);
  mineUid=uid;
  const entry={uid,id,date:Date.now(),diff:lastSummary.diffLabel,
    avgD:lastSummary.avgD,acc:lastSummary.acc,H:lastSummary.H,F:lastSummary.F};
  const b=loadBoard(); b.push(entry); saveBoard(b);
  submitted=true;
  const rank=sortedBoard(b).findIndex(e=>e.uid===uid)+1;
  const btn=$('btn-submit'); btn.disabled=true; btn.textContent=`✓ 已提交 · 当前第 ${rank} 名`;
  openBoard('screen-results');
}

/* ---------------- 事件绑定 ---------------- */
$('btn-start').addEventListener('click',startExperiment);
$('btn-begin-block').addEventListener('click',beginBlock);
$('btn-signal').addEventListener('click',()=>respond(true));
$('btn-noise').addEventListener('click',()=>respond(false));
$('btn-next').addEventListener('click',advance);
$('btn-restart').addEventListener('click',()=>location.reload());
$('btn-csv').addEventListener('click',exportCSV);
$('btn-submit').addEventListener('click',submitScore);
$('btn-board').addEventListener('click',()=>openBoard('screen-welcome'));
$('btn-board-2').addEventListener('click',()=>openBoard('screen-results'));
$('btn-board-back').addEventListener('click',()=>show(boardReturnTo));
// 记住自定义 ID
try{ const savedId=localStorage.getItem(ID_KEY); if(savedId) $('inp-id').value=savedId; }catch(e){}
$('inp-id').addEventListener('input',e=>{
  try{ localStorage.setItem(ID_KEY,e.target.value.slice(0,12)); }catch(err){}
});

document.addEventListener('keydown',e=>{
  const onTrial=$('screen-trial').classList.contains('active');
  const onIntro=$('screen-blockintro').classList.contains('active');
  const onWelcome=$('screen-welcome').classList.contains('active');
  if(e.key===' '){ e.preventDefault();
    if(onTrial && phase==='feedback') advance();
    else if(onIntro) beginBlock();
    return;
  }
  if(onTrial && phase==='ready'){
    if(e.key==='f'||e.key==='F'||e.key==='ArrowLeft'){ e.preventDefault(); respond(false); }
    if(e.key==='j'||e.key==='J'||e.key==='ArrowRight'){ e.preventDefault(); respond(true); }
  }
  if(onWelcome && e.key==='Enter'){ e.preventDefault(); startExperiment(); }
  if(onIntro && e.key==='Enter'){ e.preventDefault(); beginBlock(); }
});

drawDemos();
