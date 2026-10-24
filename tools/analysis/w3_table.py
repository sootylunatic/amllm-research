import json, glob, statistics as st
T=4.303
def decide(dl,thr=0.010):
    m=st.fmean(dl); h=T*st.stdev(dl)/3**0.5; lo,hi=m-h,m+h
    if m<=-thr and hi<0 and all(x<0 for x in dl): v="SUPPORTED (better)"
    elif m>=thr and lo>0 and all(x>0 for x in dl): v="NOT_SUPPORTED_IN_SCOPE (worse)"
    else: v="INCONCLUSIVE"
    return v,m,lo,hi
R={}
for p in glob.glob('/srv/f5/backups/w[23]*/results/round_v3_w[23]/*/*_s4?/report.json'):
    rnd=p.split('/results/')[1].split('/')[0]; name=p.split(f'/{rnd}/')[1].split('/')[0]; seed=int(p.rsplit('_s',1)[1].split('/')[0])
    r=json.load(open(p)); e=r['eval']; mode='fixed' if 'fixed' in e else 'hard_surv_0.5'
    R.setdefault(name,{})[seed]=dict(val=e[mode]['val_loss'],params=r['parameters']['total'],routed=e[mode]['flops_per_token']/1e6,
        execd=e[mode]['flops_per_token_executed']/1e6,steps=r['steps_completed'],curve={c['step']:c['val_loss'] for c in r.get('curve',[])})
names=["B_moe","B_fixed","FDC1","MoE12","B3H","B1F","FDC1_18k","MoE12_18k"]
print("CONFIG|n|steps|mean|s42/s43/s44|params M|routed MFLOP|exec MFLOP (bmm)")
for n in names:
    v=R.get(n,{}); s=sorted(v)
    if len(v)<3: print(n,"INCOMPLETE",len(v)); continue
    f=lambda k: st.fmean(x[k] for x in v.values())
    print(f"{n}|{len(v)}|{v[s[0]]['steps']}|{f('val'):.4f}|{'/'.join(f'{v[k]['val']:.4f}' for k in s)}|{v[s[0]]['params']/1e6:.2f}|{f('routed'):.2f}|{f('execd'):.2f}")
def pair(a,b,label):
    if len(R.get(a,{}))<3 or len(R.get(b,{}))<3: print(label,"|INCOMPLETE"); return
    dl=[R[a][s]['val']-R[b][s]['val'] for s in (42,43,44)]; d,m,lo,hi=decide(dl)
    print(f"{label}|{m:+.4f}|[{lo:+.4f}; {hi:+.4f}]|{d}|seeds {' '.join(f'{x:+.4f}' for x in dl)}")
print("PAIRS")
pair("FDC1","MoE12","PRIMARY FDC1 − MoE12")
pair("FDC1","B3H","FDC1 − B3H (flat routing, 3 banks, equal params)")
pair("B1F","B_fixed","B1F − B_fixed (flat routing, 1 bank, equal params)")
pair("B3H","B_fixed","B3H − B_fixed (bank count, hierarchical; 1.83x params)")
pair("FDC1","B1F","FDC1 − B1F (bank count, flat; 1.83x params)")
pair("MoE12","B_moe","MoE12 − B_moe (descriptive)")
pair("FDC1_18k","MoE12_18k","FDC1_18k − MoE12_18k")
pair("FDC1_18k","FDC1","FDC1_18k − FDC1 (descriptive)")
pair("MoE12_18k","MoE12","MoE12_18k − MoE12 (descriptive)")
if all(len(R.get(n,{}))==3 for n in ("FDC1","B3H","B1F","B_fixed")):
    inter=[(R["FDC1"][s]['val']-R["B3H"][s]['val'])-(R["B1F"][s]['val']-R["B_fixed"][s]['val']) for s in (42,43,44)]
    d,m,lo,hi=decide(inter); print(f"INTERACTION (FDC1−B3H)−(B1F−B_fixed)|{m:+.4f}|[{lo:+.4f}; {hi:+.4f}]|descriptive")
mean_d=st.fmean(R["FDC1"][s]['val']-R["MoE12"][s]['val'] for s in (42,43,44))
print(f"W3b CONDITION (mean FDC1−MoE12 <= +0.010): {mean_d:+.4f} -> {'MET' if mean_d<=0.010 else 'NOT MET'}")
print("CURVES (val at steps, mean of seeds)")
for n in ("FDC1_18k","MoE12_18k"):
    v=R.get(n,{})
    if len(v)==3:
        steps=sorted(set.intersection(*[set(x['curve']) for x in v.values()]))
        print(n, " ".join(f"{s}:{st.fmean(x['curve'][s] for x in v.values()):.4f}" for s in steps))
