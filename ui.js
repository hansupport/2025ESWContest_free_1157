function mmToCm(x){ if(x==null) return "—"; return (x/10).toFixed(1); }
function tsFmt(t){ if(!t) return "—"; const d = new Date(t*1000); return d.toLocaleString(); }
function setText(id, txt){ const el=document.getElementById(id); if(el) el.textContent = txt; }
function setWarn(msg){
  const el = document.getElementById('warn');
  if(!el) return;
  if(msg && msg.length){
    el.textContent = msg;
    el.style.display = 'block';
  }else{
    el.textContent = '';
    el.style.display = 'none';
  }
}
function setLamps(status){
  const ids = ["analyzing","done","paused","error"];
  ids.forEach(k=>{
    const el = document.getElementById("lamp_"+k);
    if(!el) return;
    if(status===k) el.classList.add("on"); else el.classList.remove("on");
  });
}

/* ====== 숫자/단위를 분리 렌더 ====== */
function setMmHTML(id, mmVal){
  const el = document.getElementById(id);
  if(!el) return;
  if(mmVal == null){
    el.textContent = '—';
  }else{
    el.innerHTML = `<span class="num">${mmVal}</span><span class="unit"> mm</span>`;
  }
}
function setCmHTML(id, mmVal){
  const el = document.getElementById(id);
  if(!el) return;
  if(mmVal == null){
    el.textContent = '—';
  }else{
    el.innerHTML = `<span class="num">${mmToCm(mmVal)}</span><span class="unit"> cm</span>`;
  }
}

// 기록(최신 4개)
let __prev_ts = null;
let __history = [];
function pushHistory(s) {
  const name = s.type_name || '—';
  const w = (s.W!=null ? s.W : '—');
  const l = (s.L!=null ? s.L : '—');
  const h = (s.H!=null ? s.H : '—');
  const bub = (s.bubble_mm!=null ? s.bubble_mm : '—');
  const pv = (s.p==null) ? '—' : (typeof s.p==='number' ? s.p.toFixed(3) : String(s.p));
  const ts = s.updated_ts ? new Date(s.updated_ts*1000).toLocaleTimeString() : '—';
  const item = { name, w, l, h, bub, pv, ts, type_id: s.type_id || null };
  __history.unshift(item);
  if (__history.length > 4) __history.pop();

  const ul = document.getElementById('history');
  if (!ul) return;
  ul.innerHTML = '';
  __history.forEach((it) => {
    const li = document.createElement('li');
    /* 히스토리는 기존처럼 텍스트로 표시.
       여기서도 단위 작게 하고 싶다면 span.unit을 써서 innerHTML로 구성하면 됨. */
    li.innerHTML = `
      <span class="left">${it.name} (${it.w}/${it.l}/${it.h}mm)</span>
      <span class="right">${it.bub}mm • p=${it.pv} • ${it.ts}</span>
    `;
    ul.appendChild(li);
  });
}

async function tick(){
  try{
    const r = await fetch('/api/state', {cache:'no-store'});
    const s = await r.json();

    // 종류
    setText('type_name', s.type_name || '—');
    setText('type_id', s.type_id ? ('ID: '+s.type_id) : '—');

    // 치수 (단위 분리)
    setMmHTML('w', s.W);
    setMmHTML('l', s.L);
    setMmHTML('h', s.H);
    setCmHTML('w_cm', s.W);
    setCmHTML('l_cm', s.L);
    setCmHTML('h_cm', s.H);

    // 뽁뽁이 길이 (단위 분리)
    setMmHTML('bubble', s.bubble_mm);
    setCmHTML('bubble_cm', s.bubble_mm);

    // 경고/확률
    setWarn(s.warn_msg);
    const pv = (s.p==null) ? '—' : (typeof s.p==='number' ? s.p.toFixed(3) : String(s.p));
    setText('prob', 'p = ' + pv);

    // 공통
    setText('ts', tsFmt(s.updated_ts));
    const p = s.params || {};
    const paramsEl = document.getElementById('params');
    if (paramsEl) {
      paramsEl.textContent =
        `계산식: 2×(W+L)×${p.layers||'?'}층 + overlap(${p.overlap_mm||'?'}mm) + slack(${Math.round((p.slack_ratio||0)*100)}%)`;
    }

    // 상태 램프
    setLamps(s.status || null);

    // 좌측 이미지 (type_id 기준)
    const imgEl = document.getElementById('item_img');
    if (imgEl) {
      if (s.type_id) {
        imgEl.src = '/imgfile/' + s.type_id + '.jpg';
        imgEl.alt = s.type_name || s.type_id;
      } else {
        imgEl.removeAttribute('src');
      }
    }

    // 히스토리 업데이트
    if (s.updated_ts && s.updated_ts !== __prev_ts) {
      if (s.type_name || (s.W!=null && s.L!=null && s.H!=null) || s.bubble_mm!=null) {
        pushHistory(s);
      }
      __prev_ts = s.updated_ts;
    }
  }catch(e){ /* noop */ }
}

document.addEventListener('DOMContentLoaded', () => {
  tick();
  setInterval(tick, 500);
});
