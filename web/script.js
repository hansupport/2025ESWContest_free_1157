/* ui.js — 안정화 버전 (사파리 호환/중복키 개선/캐시 버스터)
   - 완료/오류 상태일 때만 기록
   - W/L/H/bubble 모두 준비됐을 때만 기록
   - dedupe 기준: event_id(있으면) → updated_ts(초/밀리초 자동 판별)
*/
const ROLL_WIDTH_MM = 200;

function mmToCm(x){ if(x==null) return "—"; return (x/10).toFixed(1); }

// t가 초/밀리초 무엇이든 알아서 처리
function tsToDate(t){
  if(!t && t!==0) return null;
  // 밀리초 범위(>=1e12)면 그대로, 아니면 초→밀리초
  var ms = (t >= 1e12) ? t : (t * 1000);
  try { return new Date(ms); } catch(e){ return null; }
}
function tsFmt(t){
  var d = tsToDate(t);
  return d ? d.toLocaleString() : "—";
}

function setText(id, txt){
  var el = document.getElementById(id);
  if(el) el.textContent = txt;
}
function setWarn(msg){
  var el = document.getElementById('warn');
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
  var ids = ["analyzing","done","paused","error"];
  for(var i=0;i<ids.length;i++){
    var k = ids[i];
    var el = document.getElementById("lamp_"+k);
    if(!el) continue;
    if(status===k) el.classList.add("on"); else el.classList.remove("on");
  }
}

// 숫자/단위 분리 렌더
function setMmHTML(id, mmVal){
  var el = document.getElementById(id);
  if(!el) return;
  if(mmVal == null){
    el.textContent = '—';
  }else{
    el.innerHTML = '<span class="num">'+mmVal+'</span><span class="unit"> mm</span>';
  }
}
function setCmHTML(id, mmVal){
  var el = document.getElementById(id);
  if(!el) return;
  if(mmVal == null){
    el.textContent = '—';
  }else{
    el.innerHTML = '<span class="num">'+mmToCm(mmVal)+'</span><span class="unit"> cm</span>';
  }
}

/* ====== 기록(최신 4개) ====== */
var __prev_ts = null;
var __history = [];
var __last_history_key = null;  // 마지막으로 기록한 식별자(event_id/updated_ts)

function pushHistory(s) {
  var name = s.type_name || '—';
  var w = (s.W!=null ? s.W : '—');
  var l = (s.L!=null ? s.L : '—');
  var h = (s.H!=null ? s.H : '—');
  var bub = (s.bubble_mm!=null ? s.bubble_mm : '—');
  var pv = (s.p==null) ? '—' : (typeof s.p==='number' ? s.p.toFixed(3) : String(s.p));
  var d = tsToDate(s.updated_ts);
  var ts = d ? d.toLocaleTimeString() : '—';
  var item = { name:name, w:w, l:l, h:h, bub:bub, pv:pv, ts:ts, type_id: s.type_id || null };

  __history.unshift(item);
  if (__history.length > 4) __history.pop();

  var ul = document.getElementById('history');
  if (!ul) return;
  ul.innerHTML = '';
  for (var i=0; i<__history.length; i++){
    var it = __history[i];
    var li = document.createElement('li');
    li.innerHTML =
      '<span class="left">'+it.name+' ('+it.w+'/'+it.l+'/'+it.h+'mm)</span>' +
      '<span class="right">'+it.bub+'mm • p='+it.pv+' • '+it.ts+'</span>';
    ul.appendChild(li);
  }
}

async function tick(){
  try{
    // 캐시 버스터(강화)
    var r = await fetch('/api/state?t=' + Date.now(), {
      cache:'no-store',
      headers: { 'Cache-Control': 'no-store' }
    });
    var s = await r.json();

    // 종류
    setText('type_name', s.type_name || '—');
    setText('type_id', s.type_id ? ('ID: '+s.type_id) : '—');

    // 치수
    setMmHTML('w', s.W);
    setMmHTML('l', s.L);
    setMmHTML('h', s.H);
    setCmHTML('w_cm', s.W);
    setCmHTML('l_cm', s.L);
    setCmHTML('h_cm', s.H);

    // 뽁뽁이 길이
    setMmHTML('bubble', s.bubble_mm);
    setCmHTML('bubble_cm', s.bubble_mm);

    // 경고/확률
    setWarn(s.warn_msg);
    var pv = (s.p==null) ? '—' : (typeof s.p==='number' ? s.p.toFixed(3) : String(s.p));
    setText('prob', 'p = ' + pv);

    // 공통
    setText('ts', tsFmt(s.updated_ts));
    var p = s.params || {};
    var layers = (p.layers!=null ? p.layers : '?');
    var overlap = (p.overlap_mm!=null ? p.overlap_mm : '?');
    var slack_ratio = (p.slack_ratio!=null ? p.slack_ratio : 0);
    var slackPct = Math.round(slack_ratio * 100);

    var paramsEl = document.getElementById('params');
    if (paramsEl) {
      paramsEl.classList.add('small','mono'); // 작게 + 고정폭
      paramsEl.textContent =
'계산식 (방향 최적화, 롤폭 '+ROLL_WIDTH_MM+'mm)\n' +
'한 바퀴 = 2×(두 변의 합) / 줄수 = ceil(덮는 축 / '+ROLL_WIDTH_MM+'mm)\n' +
'총 = 한 바퀴×'+layers+'층×줄수 + overlap('+overlap+'mm) → 마지막에 slack('+slackPct+'%) 적용';
    }

    // (옵션) 라벨 모드 안내
    var params2 = document.getElementById('params2');
    if (params2) {
      params2.classList.add('small','mono');
      var labelLike = (s.type_id && /^\d{16}$/.test(s.type_id));
      if (labelLike) {
        var wrapLayers = parseInt(s.type_id[15], 10);
        params2.style.display = 'block';
        params2.textContent =
'라벨 모드\n' +
'감싸는 횟수 = '+wrapLayers+'층\n' +
'총 = 둘레×'+wrapLayers+'×줄수 (overlap/slack 제외)';
      } else {
        params2.style.display = 'none';
        params2.textContent = '';
      }
    }

    // 상태 램프
    setLamps(s.status || null);

    // 좌측 이미지 + 캐시 버스터
    var imgEl = document.getElementById('item_img');
    if (imgEl) {
      if (s.type_id) {
        var bust = s.updated_ts ? ('?v='+s.updated_ts) : '';
        imgEl.src = '/imgfile/' + s.type_id + '.jpg' + bust;
        imgEl.alt = s.type_name || s.type_id;
      } else {
        imgEl.removeAttribute('src');
      }
    }

    /* ====== 히스토리 업데이트 ======
       1) 상태 done/error
       2) W/L/H/bubble_mm 값 준비
       3) event_id(있으면) 또는 updated_ts(초/밀리초)로 dedupe
    */
    var readyForHistory = (s.status === 'done' || s.status === 'error')
      && (s.W != null && s.L != null && s.H != null && s.bubble_mm != null);

    if (readyForHistory) {
      var idPart = (s.event_id!=null ? s.event_id : (s.updated_ts!=null ? s.updated_ts : 0));
      var key = String(idPart); // 동일 이벤트면 동일키, 새 이벤트면 다른키
      if (key !== __last_history_key) {
        pushHistory(s);
        __last_history_key = key;
      }
    }

    // 화면 갱신 기준(참고용)
    if (s.updated_ts && s.updated_ts !== __prev_ts) {
      __prev_ts = s.updated_ts;
    }

  }catch(e){
    // 디버깅에 도움되도록 콘솔 출력
    if (window && window.console && console.error) {
      console.error('[ui] tick error:', e);
    }
  }
}

document.addEventListener('DOMContentLoaded', function(){
  tick();
  setInterval(tick, 500);
});
