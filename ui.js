/* ui.js — (중복 히스토리 방지 버전)
   - 완료/오류 상태일 때만 기록
   - W/L/H/bubble이 모두 준비됐을 때만 기록
   - 동일 샘플 키(상태|type_id|W|L|H|bubble)가 같으면 한 번만 기록
*/
const ROLL_WIDTH_MM = 200;  // 백엔드의 ROLL_WIDTH_MM(기본 200mm)과 맞춰두기

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

/* ====== 기록(최신 4개) ====== */
let __prev_ts = null;
let __history = [];
let __last_history_key = null; // 마지막으로 기록한 샘플의 고유 키

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
    /* 히스토리는 기존처럼 텍스트로 표시. */
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
    const slackPct = Math.round((p.slack_ratio || 0) * 100);
    const paramsEl = document.getElementById('params');
    if (paramsEl) {
      paramsEl.classList.add('small','mono'); // 작게 + 고정폭
      paramsEl.textContent =
    `계산식 (방향 최적화, 롤폭 ${ROLL_WIDTH_MM}mm)
    한 바퀴 = 2×(두 변의 합) / 줄수 = ceil(덮는 축 / ${ROLL_WIDTH_MM}mm) 
    총 = 한 바퀴×${p.layers ?? '?'}층×줄수 + overlap(${p.overlap_mm ?? '?'}mm) → 마지막에 slack(${slackPct}%) 적용`;
    }


    // (추가) 라벨 모드 안내: 16자리일 때 마지막 자리=감싸는 횟수 사용
    const params2 = document.getElementById('params2');
    if (params2) {
      params2.classList.add('small','mono');
      const labelLike = (s.type_id && /^\d{16}$/.test(s.type_id));
      if (labelLike) {
        const wrapLayers = parseInt(s.type_id[15], 10);
        params2.style.display = 'block';
        params2.textContent =
    `라벨 모드
    감싸는 횟수 = ${wrapLayers}층
    총 = 둘레×${wrapLayers}×줄수 (overlap/slack 제외)`;
      } else {
        params2.style.display = 'none';
        params2.textContent = '';
      }
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

    /* ====== 히스토리 업데이트(중복 방지) ======
       조건:
       1) 상태가 done/error
       2) W/L/H/bubble_mm 값이 모두 준비
       3) 직전에 기록한 키와 다름
    */
    const readyForHistory = (s.status === 'done' || s.status === 'error')
      && (s.W != null && s.L != null && s.H != null && s.bubble_mm != null);

    if (readyForHistory) {
      const key = [
        s.status || '',
        s.type_id || '',
        s.W ?? '',
        s.L ?? '',
        s.H ?? '',
        s.bubble_mm ?? ''
      ].join('|');

      if (key !== __last_history_key) {
        pushHistory(s);
        __last_history_key = key;
      }
    }

    // 화면 갱신용 기준값(히스토리 트리거에는 사용하지 않음)
    if (s.updated_ts && s.updated_ts !== __prev_ts) {
      __prev_ts = s.updated_ts;
    }

  }catch(e){
    // 네트워크 오류 등은 무시
  }
}

document.addEventListener('DOMContentLoaded', () => {
  tick();
  setInterval(tick, 500);
});
