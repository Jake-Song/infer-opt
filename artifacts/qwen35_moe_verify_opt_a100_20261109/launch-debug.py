import subprocess,sys,os
os.environ["TRITON_CACHE_DIR"]="/tmp/infer-opt-models/q35moe-verify-opt-20261109/triton"
log=open("/content/q35moe-verify-opt-20261109/debug-reduction.log","w")
p=subprocess.Popen([sys.executable,"/content/q35moe-verify-opt-20261109/debug-reduction.py"],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
