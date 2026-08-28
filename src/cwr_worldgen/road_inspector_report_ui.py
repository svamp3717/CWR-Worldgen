# SPDX-License-Identifier: GPL-3.0-or-later
"""Add practical filtering and metric details to the inspector HTML report."""
from __future__ import annotations

from . import road_inspector as _core


_ORIGINAL_HTML_DOCUMENT = None
_INSTALLED = False

_UI_CSS = r"""
#issue-search { min-width:210px; flex:1; }
.issue details { margin-top:7px; color:#bbb; }
.issue details pre { white-space:pre-wrap; word-break:break-word; margin:5px 0 0; font-size:11px; }
.source-context { margin-top:5px; color:#9dc6a7; font-family:ui-monospace,Consolas,monospace; font-size:12px; }
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

  function searchable(i){
    return [i.issue_id,i.severity,i.category,i.x,i.z,(i.object_ids||[]).join(' '),(i.models||[]).join(' '),i.message,i.candidate_fix,JSON.stringify(i.metrics||{})].join(' ').toLowerCase();
  }
  const textById=new Map(issues.map(i=>[i.issue_id,searchable(i)]));

  function decorateRows(){
    for(const row of list.querySelectorAll('.issue')){
      if(row.dataset.inspectorDecorated==='1') continue;
      row.dataset.inspectorDecorated='1';
      const issue=byId.get(row.dataset.row);
      if(!issue) continue;
      const metrics=issue.metrics||{};
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
    for(const row of list.querySelectorAll('.issue')){
      const text=textById.get(row.dataset.row)||'';
      row.style.display=(!query||text.includes(query))?'':'none';
    }
  }

  const observer=new MutationObserver(decorateRows);
  observer.observe(list,{childList:true});
  search.addEventListener('input',applySearch);
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
