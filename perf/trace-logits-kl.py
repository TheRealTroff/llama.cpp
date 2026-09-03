# trace-logits-kl.py <a.bin> <b.bin> [tokA tokB] : KL / mean|d| / max|d| between two raw f32 logit vectors
# (MSR_DUMP_LOGITS from llama-multiseq-repro); optional pair of token ids prints the logit margin a-b
import sys, numpy as np
a=np.fromfile(sys.argv[1],dtype=np.float32); b=np.fromfile(sys.argv[2],dtype=np.float32)
def kl(x,y):
    px=np.exp(x-x.max()); px/=px.sum(); py=np.exp(y-y.max()); py/=py.sum(); return float((px*(np.log(px+1e-30)-np.log(py+1e-30))).sum())
d=np.abs(a-b); print(f"KL {kl(a,b):.5f} mean|d| {d.mean():.4f} max|d| {d.max():.3f} argmax {a.argmax()} / {b.argmax()}")
if len(sys.argv)>4:
    i,j=int(sys.argv[3]),int(sys.argv[4]); print(f"margin[{i}-{j}] A {a[i]-a[j]:.4f}  B {b[i]-b[j]:.4f}")
