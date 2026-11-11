"""Check C++/Python source-tensor generation agrees before comparing timings."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def main():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('moe_compare', here/'compare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = (here/'bench.inc').read_text()
    functions = source[source.index('static uint32_t infer_hash'):source.index('static void infer_fill')]
    fixtures = [(903,False,0),(901,True,0),(901,True,255*1024*2048),(902,True,255*2048*512)]
    main = 'int main() { float data[257];\n'
    for salt, weight, offset in fixtures:
        main += f'for (uint32_t i=0;i<257;++i) data[i]=infer_value({offset}U+i,{salt}U,{str(weight).lower()}); fwrite(data,sizeof(float),257,stdout);\n'
    main += '}\n'
    with tempfile.TemporaryDirectory(prefix='infer-moe-fixture-') as tmp:
        cpp, exe = Path(tmp)/'fixture.cpp', Path(tmp)/'fixture'
        cpp.write_text('#include <cstdint>\n#include <cstdio>\n'+functions+main)
        subprocess.run(['g++','-std=c++17','-O2',str(cpp),'-o',str(exe)],check=True)
        actual = np.frombuffer(subprocess.check_output([str(exe)]),dtype=np.float32)
    expected = np.concatenate([module.synth(257,salt,weight,offset) for salt,weight,offset in fixtures])
    assert np.array_equal(actual,expected)
    # Check the actual native-layout index conversion against independent GU indexing.
    mapping = source[source.index('            uint32_t index = uint32_t(row * k + i);'):source.index('            values[i] = infer_value')]
    mapping = mapping.replace('uint32_t index =', 'index =')
    points = [(up,e,r,c) for up in [False,True] for e in [0,1,127,255]
              for r in [0,1,511] for c in [0,31,127,2047]]
    program = '#include <cstdint>\n#include <cstdio>\n'+functions+'int main() {\n'
    for up,e,r,c in points:
        program += '{ uint32_t index; const int64_t row='+str(e*512+r)+',k=2048,i='+str(c)+'; const bool native_up='+str(up).lower()+',native_gate='+str(not up).lower()+';\n'
        program += mapping+'float v=infer_value(index,901U,true);fwrite(&v,sizeof(v),1,stdout);}\n'
    program += '}\n'
    with tempfile.TemporaryDirectory(prefix='infer-moe-layout-fixture-') as tmp:
        cpp,exe=Path(tmp)/'fixture.cpp',Path(tmp)/'fixture'
        cpp.write_text(program)
        subprocess.run(['g++','-std=c++17','-O2',str(cpp),'-o',str(exe)],check=True)
        mapped=np.frombuffer(subprocess.check_output([str(exe)]),dtype=np.float32)
    view_values = np.array([module.synth(1,901,True,int(np.ravel_multi_index((e,r+512*up,c),(256,1024,2048))))[0]
                           for up,e,r,c in points],dtype=np.float32)
    assert np.array_equal(mapped,view_values)
    print(json.dumps({'cpp_python_fixture_values_equal':int(actual.size),'fixtures':len(fixtures),
                      'native_vs_view_source_values_equal':len(points)}))


if __name__=='__main__':
    main()
