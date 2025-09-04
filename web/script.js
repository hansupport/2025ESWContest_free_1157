/*
Web UI 스크립트
500ms마다 /api/state 폴링 → DOM 갱신
최신 캡처 이미지 표시: "__mem__"→/api/capture.jpg, 파일명→/imgfile/* (캐시 버스터 적용)
히스토리 최근 4건 유지
*/
const ROLL_WIDTH_MM = 200;

// 박스 규격(mm)
// 1호: 13×10×10cm → [130,100,100]
// 2호: 13×13×10cm → [130,130,100]
const BOX1 = { name: "1호", dims: [120, 100, 100] };
const BOX2 = { name: "2호", dims: [130, 130, 100] };

// 뽁뽁이 두께 
let LAYER_THICK_MM = 0.4; // 0.4 mm

function mmToCm(x){ if(x==null) return "—"; return (x/10).toFixed(1); }

// mm-> cm 변환(소숫점 1자리)
function mmToCm(x){ if(x==null) return "—"; return (x/10).toFixed(1); }

// 타임스탬프 초/밀리초 자동 판별
function tsToDate(t){
  if(!t && t!==0) return null;
  // 밀리초 범위(>=1e12)면 그대로, 아니면 초→밀리초
  var ms = (t >= 1e12) ? t : (t * 1000);
  try { return new Date(ms); } catch(e){ return null; }
}

//  date 포맷터
function tsFmt(t){
  var d = tsToDate(t);
  return d ? d.toLocaleString() : "—";
}

// 안전한 텍스트 주입
function setText(id, txt){
  var el = document.getElementById(id);
  if(el) el.textContent = txt;
}

// 경고 배너 토글, 빈 값이면 감춤
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

// 상단 상태 램프 on/off. status : 분석중, 분석 완료, 에러, 일시정지
function setLamps(status){
  var ids = ["analyzing","done","paused","error"];
  for(var i=0;i<ids.length;i++){
    var k = ids[i];
    var el = document.getElementById("lamp_"+k);
    if(!el) continue;
    if(status===k) el.classList.add("on"); else el.classList.remove("on");
  }
}

// 숫자/단위 분리 렌더 -mm
function setMmHTML(id, mmVal){
  var el = document.getElementById(id);
  if(!el) return;
  if(mmVal == null){
    el.textContent = '—';
  }else{
    el.innerHTML = '<span class="num">'+mmVal+'</span><span class="unit"> mm</span>';
  }
}
// 숫자/단위 분리 렌더 -cm
function setCmHTML(id, mmVal){
  var el = document.getElementById(id);
  if(!el) return;
  if(mmVal == null){
    el.textContent = '—';
  }else{
    el.innerHTML = '<span class="num">'+mmToCm(mmVal)+'</span><span class="unit"> cm</span>';
  }
}

// 겹수 해석 우선순위
function resolveLayers(s){
  if (typeof s.wrap_layers === 'number' && s.wrap_layers > 0) return s.wrap_layers;

  if (s.type_id && /^\d{16}$/.test(s.type_id)) {
    var d = parseInt(s.type_id[15], 10);
    if (!isNaN(d) && d > 0) return d;
  }

  if (s.params && s.params.layers != null && !isNaN(Number(s.params.layers))) {
    var n = Number(s.params.layers);
    if (n > 0) return n;
  }

  return 1;
}

// 박스 선택 로직
// 각 변에 pad = 2 * layers * LAYER_THICK_MM (mm) 패딩
// 물체와 박스 모두 오름차순 정렬 후 성분별 비교
function pickBox(mmDims, layers){
  var pad = 2 * (layers > 0 ? layers : 1) * LAYER_THICK_MM; // mm
  var obj = [mmDims.w + pad, mmDims.l + pad, mmDims.h + pad].sort((a,b)=>a-b);

  function fits(box){
    var bx = box.dims.slice().sort((a,b)=>a-b);
    return obj[0] <= bx[0] && obj[1] <= bx[1] && obj[2] <= bx[2];
  }
  if (fits(BOX1)) return BOX1.name;
  if (fits(BOX2)) return BOX2.name;
  return "적합 없음";
}

// 기록(최신 4개)/ 완료/에러 상태에서만 기록. 
var __prev_ts = null;
var __history = [];
var __last_history_key = null;

// 히스토리에 1건 추가하고 리스트 렌더
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
// 메인 폴링 루프(500ms) (최신 상태/이미지를 즉시 반영.)
async function tick(){
  try{
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

    // 공통 정보/ 계산식 안내
    setText('ts', tsFmt(s.updated_ts));

    // 라벨 모드 안내
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

    // 상태 램프 갱신
    setLamps(s.status || null);

    // 좌측 이미지 + 캐시 버스터
    var imgEl = document.getElementById('item_img');
    if (imgEl) {
      var bust = s.updated_ts ? ('?v='+s.updated_ts) : '';
      var src = null;

      if (s.cap_image) {
        if (s.cap_image === '__mem__') {
          // 메모리에서 바로 서빙되는 최신 캡처
          src = '/api/capture.jpg' + bust;
          imgEl.alt = (s.type_name ? (s.type_name + ' (캡처)') : 'capture');
        } else {
          // 예전 방식(파일명 제공 시)
          src = '/imgfile/' + s.cap_image + bust;
          imgEl.alt = s.type_name || s.cap_image;
        }
      } else if (s.type_id) {
        // 타입별 정적 이미지(없으면 브라우저 404)
        src = '/imgfile/' + s.type_id + '.jpg' + bust;
        imgEl.alt = s.type_name || s.type_id;
      }

      if (src) imgEl.src = src; else imgEl.removeAttribute('src');
    }

    // 히스토리 업데이트 상태 : done/error  /  W/L/H/bubble_mm 값 준비 / event_id(있으면) 또는 updated_ts(초/밀리초)로 dedupe
    var readyForHistory = (s.status === 'done' || s.status === 'error')
      && (s.W != null && s.L != null && s.H != null && s.bubble_mm != null);

    if (readyForHistory) {
      var idPart = (s.event_id!=null ? s.event_id : (s.updated_ts!=null ? s.updated_ts : 0));
      var key = String(idPart);
      if (key !== __last_history_key) {
        pushHistory(s);
        __last_history_key = key;
      }
    }

    // 화면 갱신 기준
    if (s.updated_ts && s.updated_ts !== __prev_ts) {
      __prev_ts = s.updated_ts;
    }

  }catch(e){
    // 콘솔 출력
    if (window && window.console && console.error) {
      console.error('[ui] tick error:', e);
    }
  }
}

document.addEventListener('DOMContentLoaded', function(){
  tick();
  setInterval(tick, 500);
});
