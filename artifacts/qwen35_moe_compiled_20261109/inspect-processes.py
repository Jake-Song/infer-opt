import json,time,os
from pathlib import Path
rows=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  raw=(p/'stat').read_text();tail=raw[raw.rfind(')')+2:].split()
  rows.append({'pid':int(p.name),'ppid':int(tail[1]),'name':raw[raw.find('(')+1:raw.rfind(')')],'state':tail[0],'cpu_ticks':int(tail[11])+int(tail[12]),'rss_mib':int(tail[21])*os.sysconf('SC_PAGE_SIZE')/2**20})
 except (OSError,ValueError,IndexError):pass
owned={1109}
for _ in range(5):owned.update(r['pid'] for r in rows if r['ppid'] in owned)
result={'time':time.time(),'processes':[r for r in rows if r['pid'] in owned]}
p=Path('/content/q35moe-compiled-20261109/results/process-snapshot.json');p.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
