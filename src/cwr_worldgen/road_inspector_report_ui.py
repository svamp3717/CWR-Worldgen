# SPDX-License-Identifier: GPL-3.0-or-later
"""Add practical filtering, paved focus, metric details, and CWA teleport helpers."""
from __future__ import annotations

from . import road_inspector as _core


_ORIGINAL_HTML_DOCUMENT = None
_INSTALLED = False

_UI_CSS = r"""
#issue-search { min-width:210px; flex:1; }
.issue details { margin-top:7px; color:#bbb; }
.issue details pre { white-space:pre-wrap; word-break:break-word; margin:5px 0 0; font-size:11px; }
.source-context { margin-top:5px; color:#9dc6a7; font-family:ui-monospace,Consolas,monospace; font-size:12px; }
.surface-focus-toggle { display:inline-flex; align-items:center; gap:6px; padding:5px 8px; border:1px solid #555; border-radius:5px; background:#202020; color:#ddd; font-size:12px; }
.surface-focus-toggle input { margin:0; }
.dirt-intersection-note { margin-top:6px; color:#c9b58b; font-size:12px; }
.teleports { margin-top:9px; padding-top:8px; border-top:1px solid #383838; display:grid; gap:6px; }
.teleport-row { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
.teleport-label { min-width:88px; color:#9ec9ff; font-family:ui-monospace,Consolas,monospace; font-size:12px; }
.teleport-command { flex:1; min-width:230px; padding:5px 7px; border:1px solid #444; border-radius:4px; background:#131313; color:#d8f4d8; font-family:ui-monospace,Consolas,monospace; font-size:12px; user-select:all; }
.teleport-copy { white-space:nowrap; cursor:pointer; }
.teleport-copy.copied { border-color:#6a9d6a; color:#bff0bf; }
"""

_UI_SCRIPT = r"""
<script>
(function(){
  if(typeof issues==='undefined'||typeof list==='undefined') return;
  const controls=document.querySelector('.controls');
  if(!controls) return;
  const search=document.createElement('input');
  search.id='issue-search';
  search.type='search';
  search.placeholder='search issue, object, model, road id, coordinate';
  const reset=document.getElementById('reset');
  controls.insertBefore(search,reset||null);
  const byId=new Map(issues.map(i=>[i.issue_id,i]));
  const roadById=new Map((typeof roads==='undefined'?[]:roads).map(r=>[Number(r.object_id),r]));
  const seamCategories=new Set(['straight_miter','curve_transition','connector_gap']);
  const dirtSurfaceWords=['dirt','earth','ground','gravel','fine_gravel','compacted','unpaved','sand','mud','grass'];
  const mixedJunctionRadiusMetres=7.0;

  function surfaceTokens(issue){
    const value=String((issue.metrics||{}).source_surfaces||'').toLowerCase();
    return value.split(';').map(v=>v.trim()).filter(Boolean);
  }

  function matchesSurfaceWord(value,words){
    return words.some(word=>value===word||value.startsWith(`${word}:`));
  }

  function involvedRoads(issue){
    return (issue.object_ids||[]).map(id=>roadById.get(Number(id))).filter(Boolean);
  }

  function isNativeMixedJunction(road){
    if(String(road.kind||'').toLowerCase()!=='junction_t') return false;
    const model=String(road.model||'').replaceAll('/','\\');
    return /kr_new_(?:sil|asf|kos)_ces_t\.p3d$/i.test(model);
  }

  const nativeMixedJunctions=Array.from(roadById.values()).filter(isNativeMixedJunction);

  function nearNativeMixedJunction(issue){
    if(!seamCategories.has(issue.category)) return false;
    const x=Number(issue.x), z=Number(issue.z);
    if(!Number.isFinite(x)||!Number.isFinite(z)) return false;
    return nativeMixedJunctions.some(road=>{
      if(!Array.isArray(road.center)||road.center.length<2) return false;
      return Math.hypot(x-Number(road.center[0]),z-Number(road.center[1]))<=mixedJunctionRadiusMetres;
    });
  }

  function isDirtOrMixedFinding(issue){
    const surfaces=surfaceTokens(issue);
    if(surfaces.some(value=>matchesSurfaceWord(value,dirtSurfaceWords))) return true;

    // ``ces`` is the stock dirt/service-road family in this inspector. Paved
    // focus therefore hides every ces-only diagnostic as well as ces/paved
    // transitions, not merely intersection categories. Raw JSON/CSV stays intact.
    if(involvedRoads(issue).some(road=>String(road.family||'').toLowerCase()==='ces')) return true;

    // A paved-looking seam can still be one arm of a native sil/ces T. Lundby33
    // exposed this immediately beside the mixed junction even though the two
    // objects named by the seam were both sil. Keep that out of paved-only focus.
    if(nearNativeMixedJunction(issue)) return true;
    return false;
  }

  const nonPavedFindingIds=new Set(
    issues.filter(isDirtOrMixedFinding).map(issue=>issue.issue_id)
  );
  const dirtToggle=document.createElement('label');
  dirtToggle.className='surface-focus-toggle';
  const dirtCheckbox=document.createElement('input');
  dirtCheckbox.type='checkbox';
  dirtCheckbox.id='show-dirt-mixed-findings';
  const dirtCaption=document.createElement('span');
  dirtCaption.textContent=`Show dirt/mixed findings (${nonPavedFindingIds.size})`;
  dirtToggle.appendChild(dirtCheckbox);
  dirtToggle.appendChild(dirtCaption);
  controls.insertBefore(dirtToggle,reset||null);

  function searchable(i){
    const scope=nonPavedFindingIds.has(i.issue_id)?'dirt-or-mixed finding':'paved-only-or-general';
    return [i.issue_id,i.severity,i.category,i.x,i.z,(i.object_ids||[]).join(' '),(i.models||[]).join(' '),i.message,i.candidate_fix,scope,JSON.stringify(i.metrics||{})].join(' ').toLowerCase();
  }
  const textById=new Map(issues.map(i=>[i.issue_id,searchable(i)]));

  function coord(value){
    const number=Number(value);
    if(!Number.isFinite(number)) return '0';
    return number.toFixed(2).replace(/\.00$/,'');
  }

  function teleportCommand(x,z){
    return `player setPos [${coord(x)}, ${coord(z)}, 0]`;
  }

  function fallbackCopy(text){
    const area=document.createElement('textarea');
    area.value=text;
    area.setAttribute('readonly','');
    area.style.position='absolute';
    area.style.left='-9999px';
    document.body.appendChild(area);
    area.select();
    let copied=false;
    try{ copied=document.execCommand('copy'); }catch(_error){ copied=false; }
    area.remove();
    return copied;
  }

  async function copyTeleport(text,button){
    let copied=false;
    try{
      if(navigator.clipboard&&window.isSecureContext){
        await navigator.clipboard.writeText(text);
        copied=true;
      }
    }catch(_error){ copied=false; }
    if(!copied) copied=fallbackCopy(text);
    const old=button.textContent;
    button.textContent=copied?'Copied':'Select command';
    if(copied) button.classList.add('copied');
    window.setTimeout(()=>{button.textContent=old;button.classList.remove('copied');},1400);
  }

  function teleportRow(label,x,z){
    const command=teleportCommand(x,z);
    const row=document.createElement('div');
    row.className='teleport-row';
    const caption=document.createElement('span');
    caption.className='teleport-label';
    caption.textContent=label;
    const code=document.createElement('code');
    code.className='teleport-command';
    code.textContent=command;
    code.title='CWA/OFP debug-console command';
    const button=document.createElement('button');
    button.type='button';
    button.className='teleport-copy';
    button.textContent='Copy teleport';
    button.addEventListener('click',event=>{
      event.stopPropagation();
      copyTeleport(command,button);
    });
    code.addEventListener('click',event=>event.stopPropagation());
    row.appendChild(caption); row.appendChild(code); row.appendChild(button);
    return row;
  }

  function decorateRows(){
    for(const row of list.querySelectorAll('.issue')){
      if(row.dataset.inspectorDecorated==='1') continue;
      row.dataset.inspectorDecorated='1';
      const issue=byId.get(row.dataset.row);
      if(!issue) continue;
      const metrics=issue.metrics||{};
      if(nonPavedFindingIds.has(issue.issue_id)){
        row.dataset.dirtMixedFinding='1';
        const note=document.createElement('div');
        note.className='dirt-intersection-note';
        note.textContent='Dirt or mixed paved/dirt diagnostic · hidden by default in paved-only focus';
        row.appendChild(note);
      }
      if(metrics.source_road_ids||metrics.source_highways||metrics.source_surfaces){
        const source=document.createElement('div');
        source.className='source-context';
        const parts=[];
        if(metrics.source_road_ids) parts.push(`source ${metrics.source_road_ids}`);
        if(metrics.source_highways) parts.push(metrics.source_highways);
        if(metrics.source_surfaces) parts.push(metrics.source_surfaces);
        source.textContent=parts.join(' · ');
        row.appendChild(source);
      }

      const teleports=document.createElement('div');
      teleports.className='teleports';
      teleports.appendChild(teleportRow('Finding',issue.x,issue.z));
      for(const objectId of issue.object_ids||[]){
        const road=roadById.get(Number(objectId));
        if(!road||!Array.isArray(road.center)||road.center.length<2) continue;
        teleports.appendChild(teleportRow(`Road ${objectId}`,road.center[0],road.center[1]));
      }
      row.appendChild(teleports);

      const details=document.createElement('details');
      const summary=document.createElement('summary');
      summary.textContent='Issue metrics';
      const pre=document.createElement('pre');
      pre.textContent=JSON.stringify(metrics,null,2);
      details.appendChild(summary); details.appendChild(pre); row.appendChild(details);
    }
    applySearch();
  }

  function applySearch(){
    const query=search.value.trim().toLowerCase();
    const showDirt=dirtCheckbox.checked;
    for(const row of list.querySelectorAll('.issue')){
      const text=textById.get(row.dataset.row)||'';
      const matches=!query||text.includes(query);
      const hiddenDirt=!showDirt&&nonPavedFindingIds.has(row.dataset.row);
      row.style.display=(matches&&!hiddenDirt)?'':'none';
    }
    if(typeof svg!=='undefined'){
      for(const marker of svg.querySelectorAll('.marker')){
        marker.style.display=(!showDirt&&nonPavedFindingIds.has(marker.dataset.issue))?'none':'';
      }
    }
  }

  const observer=new MutationObserver(decorateRows);
  observer.observe(list,{childList:true});
  search.addEventListener('input',applySearch);
  dirtCheckbox.addEventListener('change',applySearch);
  decorateRows();
})();
</script>
"""


def _html_document(result) -> str:
    if _ORIGINAL_HTML_DOCUMENT is None:
        raise RuntimeError("Road Inspector report UI is not installed")
    document = _ORIGINAL_HTML_DOCUMENT(result)
    if "</style>" in document:
        document = document.replace("</style>", _UI_CSS + "\n</style>", 1)
    if "</body>" in document:
        document = document.replace("</body>", _UI_SCRIPT + "\n</body>", 1)
    return document


def install() -> None:
    global _ORIGINAL_HTML_DOCUMENT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_HTML_DOCUMENT = _core._html_document
    _core._html_document = _html_document
    _INSTALLED = True
