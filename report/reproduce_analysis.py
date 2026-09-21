import json, math, os, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, spearmanr

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
FIG=OUT/'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

paths={
'core': ROOT/'jev-results/run_20260919T201050Z_core',
'iso': ROOT/'jev-results/run_20260919T201739Z_core',
'arch_full': ROOT/'jev-results/run_20260919T202139Z_full',
'cal': ROOT/'jev-results/run_20260919T203758Z_calibration_full',
'sem': ROOT/'jev-results/run_20260919T211437Z_semantic_full',
}

def loadj(path):
    return json.loads(Path(path).read_text())

def load_raw(d):
    return [json.loads(x) for x in (d/'raw.jsonl').read_text().splitlines() if x.strip()]

def upstream_df(d):
    rows=[]
    for r in load_raw(d):
        h=r.get('response_headers') or {}
        try: up=float(h.get('x-envoy-upstream-service-time'))
        except: up=np.nan
        m=r.get('metadata') or {}
        usage=((r.get('response') or {}).get('usage') or {})
        rows.append({'case_id':r['case_id'],'experiment':r['experiment'],'upstream_ms':up,'wall_ms':r.get('latency_ms'),
                     'input_tokens':usage.get('input_tokens'),'output_tokens':usage.get('output_tokens'),**{f'meta_{k}':v for k,v in m.items()}})
    return pd.DataFrame(rows)

def bootstrap_ci(x, stat=np.mean, n=10000, seed=260919):
    a=np.asarray(pd.Series(x).dropna(),dtype=float)
    rng=np.random.default_rng(seed)
    if len(a)==0:return (np.nan,np.nan,np.nan)
    vals=np.empty(n)
    for i in range(n): vals[i]=stat(rng.choice(a,size=len(a),replace=True))
    return float(stat(a)), float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))

def wilson(k,n,alpha=.05):
    if n==0:return (np.nan,np.nan)
    z=norm.ppf(1-alpha/2); ph=k/n; den=1+z*z/n
    c=(ph+z*z/(2*n))/den
    h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return c-h,c+h

def entropy_arr_json(s):
    d=json.loads(s); p=np.array(list(d.values()),float); p=p[p>0]
    return float(-(p*np.log(p)).sum())

def max_arr_json(s):
    d=json.loads(s); return max(d.values())

# Summaries
core_summary=loadj(paths['core']/'summary.json')
cal_summary=loadj(paths['cal']/'summary.json')
sem_summary=loadj(paths['sem']/'summary.json')

# Architecture scaling
iso=upstream_df(paths['iso']); af=upstream_df(paths['arch_full']); core_raw=upstream_df(paths['core'])
# Use full isolated run as primary scaling
q=af[af.experiment=='parallel_question_scaling'].copy().sort_values('meta_question_count')
c=af[af.experiment=='choice_cardinality_scaling'].copy().sort_values('meta_cardinality')
ctx=af[af.experiment=='context_length_position'].copy().sort_values('input_tokens')
# One raw record per case for q/c but normalized duplicates don't matter here.

# Correlations
q_rho=spearmanr(q['output_tokens'],q['upstream_ms']).statistic
c_rho=spearmanr(c['output_tokens'],c['upstream_ms']).statistic
# ratios endpoints
q1=q.iloc[0]; qn=q.iloc[-1]; c1=c.iloc[0]; cn=c.iloc[-1]

# Calibration
cal=pd.read_csv(paths['cal']/'normalized.csv')
binary=cal[(cal['type']=='noul') & cal['target_probability'].notna()].copy()
# selected experiments exclude multiclass
exp_order=['prob_explicit_randomness','prob_bayes_single_signal','prob_bayes_two_signals','prob_base_rate_stress','prob_representation_invariance','prob_repeatability']
cal_stats={}
for e in exp_order:
    d=binary[binary.experiment==e]
    if len(d):
        cal_stats[e]={
            'n':len(d),
            'mae':bootstrap_ci(d.prob_abs_error),
            'bias':bootstrap_ci(d.prob_error),
            'rmse':float(np.sqrt(np.mean(d.prob_sq_error))),
            'corr':float(np.corrcoef(d.target_probability,d.p_true)[0,1]) if d.target_probability.nunique()>1 else np.nan
        }
# global exact recovery
cal_global={'n':len(binary),'mae':bootstrap_ci(binary.prob_abs_error),'bias':bootstrap_ci(binary.prob_error),'rmse':float(np.sqrt(binary.prob_sq_error.mean())), 'corr':float(np.corrcoef(binary.target_probability,binary.p_true)[0,1])}

multi=cal[cal.experiment=='prob_multiclass_bayes'].copy()
multi['target_max']=multi.target_distribution_json.apply(max_arr_json)
multi['pred_max']=multi.pred_distribution_json.apply(max_arr_json)
multi['target_entropy']=multi.target_distribution_json.apply(entropy_arr_json)
multi['pred_entropy']=multi.pred_distribution_json.apply(entropy_arr_json)
multi_stats={
'n':len(multi),'accuracy':float(multi.correct.mean()),
'tv':bootstrap_ci(multi.distribution_tv),'jsd':bootstrap_ci(multi.distribution_jsd),
'target_max':bootstrap_ci(multi.target_max),'pred_max':bootstrap_ci(multi.pred_max),
'overgap':bootstrap_ci(multi.pred_max-multi.target_max),'target_entropy':bootstrap_ci(multi.target_entropy),'pred_entropy':bootstrap_ci(multi.pred_entropy)
}
# representation invariance group ranges
rep=binary[binary.experiment=='prob_representation_invariance'].copy()
rep_ranges=rep.groupby('meta_group').p_true.agg(lambda x: x.max()-x.min()) if len(rep) else pd.Series(dtype=float)
rep_range_stats=bootstrap_ci(rep_ranges)
rep_range_max=float(rep_ranges.max()) if len(rep_ranges) else np.nan
# repeatability std by target
rept=binary[binary.experiment=='prob_repeatability'].copy()
repstd=rept.groupby('meta_target_probability').p_true.agg(['mean','std','min','max','count'])

# Semantic
sem=pd.read_csv(paths['sem']/'normalized.csv')
sem['reference_max']=sem['meta_reference_max_probability'].astype(float)
sem['jev_max']=sem[['confidence_top']].astype(float)
sem['overgap']=sem.jev_max-sem.reference_max
sem['ref_entropy']=sem.target_distribution_json.apply(entropy_arr_json)
sem['pred_entropy']=sem.pred_distribution_json.apply(entropy_arr_json)
sem['entropy_gap']=sem.pred_entropy-sem.ref_entropy
sem_acc=float(sem.correct.mean()); sem_ci=wilson(int(sem.correct.sum()),len(sem))
sem_stats={
'n':len(sem),'accuracy':sem_acc,'acc_ci':sem_ci,
'tv':bootstrap_ci(sem.distribution_tv),'jsd':bootstrap_ci(sem.distribution_jsd),
'refmax':bootstrap_ci(sem.reference_max),'jevmax':bootstrap_ci(sem.jev_max),
'overgap':bootstrap_ci(sem.overgap),'refentropy':bootstrap_ci(sem.ref_entropy),'predentropy':bootstrap_ci(sem.pred_entropy),'entropygap':bootstrap_ci(sem.entropy_gap)
}
# ambiguity bins
def ambig(row):
    x=row.reference_max
    if x<=.55:return '<=0.55'
    if x<=.70:return '0.55-0.70'
    if x<=.85:return '0.70-0.85'
    return '>0.85'
sem['ambig']=sem.apply(ambig,axis=1)
ambig_stats={}
for b,d in sem.groupby('ambig'):
    lo,hi=wilson(int(d.correct.sum()),len(d))
    ambig_stats[b]={'n':len(d),'acc':d.correct.mean(),'acc_lo':lo,'acc_hi':hi,'ref':d.reference_max.mean(),'jev':d.jev_max.mean(),'gap':d.overgap.mean(),'tv':d.distribution_tv.mean()}
# domains
by_domain={}
for dom,d in sem.groupby('meta_domain'):
    lo,hi=wilson(int(d.correct.sum()),len(d))
    by_domain[dom]={'n':len(d),'acc':d.correct.mean(),'acc_lo':lo,'acc_hi':hi,'tv':d.distribution_tv.mean(),'gap':d.overgap.mean(),'jev':d.jev_max.mean(),'ref':d.reference_max.mean()}
# noise analysis semantic adjudication only to avoid repeats
sad=sem[sem.experiment=='semantic_adjudication']
noise={}
for nu,d in sad.groupby('meta_noise_units'):
    noise[nu]={'n':len(d),'acc':d.correct.mean(),'tv':d.distribution_tv.mean(),'gap':d.overgap.mean(),'jev':d.jev_max.mean()}
# risk coverage: rank descending confidence_top, compute error of retained top coverage fractions
srt=sem.sort_values('confidence_top',ascending=False).reset_index(drop=True)
risk=[]
for cov in [.25,.5,.75,.9,.95,1.0]:
    n=max(1,int(round(cov*len(srt)))); d=srt.iloc[:n]
    risk.append((cov,n,1-d.correct.mean()))

# Figures
plt.figure(figsize=(8,5))
plt.plot(q['meta_question_count'],q['upstream_ms'],marker='o',label='questions')
plt.xscale('log',base=2); plt.xlabel('Questions in one request'); plt.ylabel('Upstream service time (ms)'); plt.title('Jev parallel-question scaling (isolated, concurrency=1)'); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(FIG/'question_scaling.png',dpi=180); plt.close()

plt.figure(figsize=(8,5))
plt.plot(c['meta_cardinality'],c['upstream_ms'],marker='o')
plt.xscale('log',base=2); plt.xlabel('Choice cardinality'); plt.ylabel('Upstream service time (ms)'); plt.title('Jev Choice-cardinality scaling (isolated, concurrency=1)'); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(FIG/'choice_scaling.png',dpi=180); plt.close()

plt.figure(figsize=(8,5))
for pos,d in ctx.groupby('meta_evidence_position'):
    plt.plot(d['input_tokens'],d['upstream_ms'],marker='o',label=str(pos))
plt.xlabel('Input tokens'); plt.ylabel('Upstream service time (ms)'); plt.title('Context-length and evidence-position scaling'); plt.legend(title='Evidence position'); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(FIG/'context_scaling.png',dpi=180); plt.close()

# Exact posterior scatter sample + y=x
plt.figure(figsize=(7,7))
plt.scatter(binary.target_probability,binary.p_true,s=7,alpha=.18)
plt.plot([0,1],[0,1],linestyle='--')
plt.xlabel('Exact target probability'); plt.ylabel('Jev Noul probability'); plt.title('Exact probability recovery — calibration_full'); plt.xlim(0,1); plt.ylim(0,1); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(FIG/'exact_probability_recovery.png',dpi=180); plt.close()

# calibration error by exp
labels=[]; vals=[]; los=[]; his=[]
for e in exp_order:
    if e in cal_stats:
        m,lo,hi=cal_stats[e]['mae']; labels.append(e.replace('prob_','').replace('_','\n')); vals.append(m); los.append(m-lo); his.append(hi-m)
plt.figure(figsize=(10,5)); x=np.arange(len(vals)); plt.bar(x,vals); plt.errorbar(x,vals,yerr=[los,his],fmt='none',capsize=3); plt.xticks(x,labels,rotation=0); plt.ylabel('Mean absolute probability error'); plt.title('Exact posterior error by experiment (95% bootstrap CI)'); plt.tight_layout(); plt.savefig(FIG/'calibration_mae.png',dpi=180); plt.close()

# semantic ref vs Jev max
plt.figure(figsize=(7,7)); plt.scatter(sem.reference_max,sem.jev_max,s=9,alpha=.18); plt.plot([0,1],[0,1],linestyle='--'); plt.xlabel('Reference modal probability'); plt.ylabel('Jev top probability'); plt.title('Semantic confidence vs authored adjudication'); plt.xlim(.3,1.02); plt.ylim(.3,1.02); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(FIG/'semantic_confidence.png',dpi=180); plt.close()

# risk coverage
plt.figure(figsize=(7,5)); plt.plot([x[0] for x in risk],[x[2] for x in risk],marker='o'); plt.xlabel('Coverage'); plt.ylabel('Modal error rate'); plt.title('Semantic selective-risk curve'); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(FIG/'semantic_risk_coverage.png',dpi=180); plt.close()

# domain gaps
D=sorted(by_domain); gaps=[by_domain[d]['gap'] for d in D]
plt.figure(figsize=(9,5)); plt.bar(np.arange(len(D)),gaps); plt.xticks(np.arange(len(D)),[d.replace('_','\n') for d in D]); plt.ylabel('Mean Jev top p - reference top p'); plt.title('Semantic overconfidence gap by domain'); plt.tight_layout(); plt.savefig(FIG/'semantic_domain_overconfidence.png',dpi=180); plt.close()

# helper formatting
pct=lambda x:f'{100*x:.2f}%'
ci=lambda t:f'{t[0]:.4f} (95% CI {t[1]:.4f}–{t[2]:.4f})'

# Build architecture bullets strings
q_lines=[]
for _,r in q.iterrows(): q_lines.append(f"- {int(r.meta_question_count):>3} questions: {int(r.input_tokens):,} input tokens, {int(r.output_tokens):,} nominal output tokens, {int(r.upstream_ms)} ms upstream")
c_lines=[]
for _,r in c.iterrows(): c_lines.append(f"- {int(r.meta_cardinality):>3} choices: {int(r.input_tokens):,} input tokens, {int(r.output_tokens):,} nominal output tokens, {int(r.upstream_ms)} ms upstream")
ctx_lines=[]
for _,r in ctx.iterrows(): ctx_lines.append(f"- {int(r.input_tokens):,} input tokens, evidence {r.meta_evidence_position}: {int(r.upstream_ms)} ms upstream")
cal_lines=[]
pretty={
'prob_explicit_randomness':'Explicit aleatoric probability',
'prob_bayes_single_signal':'Single-signal Bayes',
'prob_bayes_two_signals':'Two-signal/conflicting Bayes',
'prob_base_rate_stress':'Rare-base-rate stress',
'prob_representation_invariance':'Equivalent-representation invariance',
'prob_repeatability':'Repeated fixed-probability probes',}
for e in exp_order:
    if e in cal_stats:
        st=cal_stats[e]; m,lo,hi=st['mae']; b,blo,bhi=st['bias'];
        cal_lines.append(f"- **{pretty[e]}** (n={st['n']:,}): MAE {m:.4f} [95% bootstrap CI {lo:.4f}, {hi:.4f}], bias {b:+.4f} [{blo:+.4f}, {bhi:+.4f}], RMSE {st['rmse']:.4f}.")
ambig_lines=[]
for b in ['<=0.55','0.55-0.70','0.70-0.85','>0.85']:
    if b in ambig_stats:
        s=ambig_stats[b]; ambig_lines.append(f"- Reference modal probability **{b}** (n={s['n']:,}): accuracy {pct(s['acc'])} (Wilson 95% CI {pct(s['acc_lo'])}–{pct(s['acc_hi'])}), mean reference top p={s['ref']:.3f}, Jev top p={s['jev']:.3f}, gap={s['gap']:+.3f}, TV={s['tv']:.3f}.")
domain_lines=[]
for dom,s in sorted(by_domain.items()): domain_lines.append(f"- **{dom}** (n={s['n']:,}): accuracy {pct(s['acc'])} (95% Wilson {pct(s['acc_lo'])}–{pct(s['acc_hi'])}), TV={s['tv']:.3f}, overconfidence gap={s['gap']:+.3f}.")
noise_lines=[]
for nu,s in sorted(noise.items()): noise_lines.append(f"- noise_units={int(nu):>2} (n={s['n']:,}): accuracy {pct(s['acc'])}, TV={s['tv']:.3f}, gap={s['gap']:+.3f}, mean top p={s['jev']:.3f}.")
risk_lines=[]
for cov,n,err in risk:risk_lines.append(f"- coverage {pct(cov)} (n={n:,}): modal error {pct(err)}.")

report=rf'''# Jev / TypeSafe System One Models: Black-Box Empirical Characterization and Architecture Reconstruction

**Date:** 19 September 2026  
**Model observed:** `jev-1.13.0` behind the `jev-latest` alias  
**Study type:** Independent black-box API characterization  
**Status:** Research report; architecture conclusions are inferential, not access to proprietary internals

## Abstract

This report combines five black-box experiments performed against TypeSafe AI's Jev API: a 303-request architecture/robustness run, two isolated architecture-scaling runs, a 5,810-request exact-probability calibration study, and a 1,110-request semantic-calibration study. The evidence strongly supports TypeSafe's claim that Jev is **decision-native rather than an ordinary autoregressive text generator hidden behind structured output**. In isolated tests, increasing one request from 1 to 128 independent questions increased nominal output accounting from 24 to 2,564 tokens while upstream service time changed from 99 ms to 167 ms; increasing Choice cardinality from 2 to 255 increased nominal output accounting from 43 to 2,826 tokens while upstream service time remained in the same approximately 60–140 ms regime (136 ms at 2 choices, 113 ms at 255). These observations are inconsistent with interpreting the reported output-token counts as sequentially decoded tokens and are consistent with shared state computation plus vectorized/parallel decision scoring.

The calibration results are more nuanced. Jev accurately represents direct aleatoric probabilities, but exact Bayesian posterior recovery shows systematic error: across binary exact-probability cases, mean absolute probability error is {cal_global['mae'][0]:.4f} (95% bootstrap CI {cal_global['mae'][1]:.4f}–{cal_global['mae'][2]:.4f}), RMSE {cal_global['rmse']:.4f}, and Pearson correlation {cal_global['corr']:.3f}. Rare-base-rate and representation-invariance probes are substantially worse. On 1,000 multiclass Bayesian cases, Jev's mean maximum probability is {multi_stats['pred_max'][0]:.3f} versus an exact posterior mean maximum of {multi_stats['target_max'][0]:.3f}; the mean overconfidence gap is {multi_stats['overgap'][0]:+.3f} (95% bootstrap CI {multi_stats['overgap'][1]:+.3f}–{multi_stats['overgap'][2]:+.3f}).

The 1,110-case semantic study confirms the same sharpness tendency outside explicit probability arithmetic: modal accuracy is {pct(sem_acc)} (Wilson 95% CI {pct(sem_ci[0])}–{pct(sem_ci[1])}), but mean Jev top probability is {sem_stats['jevmax'][0]:.3f} versus an authored adjudication reference mean of {sem_stats['refmax'][0]:.3f}. The paired overconfidence gap is {sem_stats['overgap'][0]:+.3f} (95% bootstrap CI {sem_stats['overgap'][1]:+.3f}–{sem_stats['overgap'][2]:+.3f}). Confidence nevertheless ranks semantic risk usefully: high-confidence subsets exhibit markedly lower modal error.

The best-supported architectural hypothesis is therefore a **shared semantic state representation with dynamically represented questions/candidates and vectorized typed decision heads or scorers**, possibly with a special two-stage path for high-cardinality Choice as TypeSafe itself states. The exact backbone, parameter count, attention topology, dense/MoE status, and RLCD objective are not identifiable from the API data.

---

## 1. Research questions

The study was designed to test four distinct claims rather than conflating them:

1. **Computation:** Does Jev behave like sequential autoregressive generation, or like parallel decision computation?
2. **Runtime programmability:** Can it interpret arbitrary question/candidate semantics rather than selecting among fixed trained labels?
3. **Probabilistic validity:** Do returned numbers behave like calibrated probabilities when the correct posterior is known analytically?
4. **Semantic uncertainty:** Does confidence track ambiguity and error on realistic decision-shaped tasks?

The report uses three evidence labels throughout:

- **Observed:** directly measured from API responses or response headers.
- **Supported inference:** the observation strongly favors an explanation but does not uniquely identify it.
- **Hypothesis:** plausible reconstruction that remains underdetermined by black-box evidence.

This distinction is essential: black-box timing can falsify some architectures, but it cannot prove the exact internal network graph.

---

## 2. TypeSafe's public claims being tested

TypeSafe's documentation says Jev accepts unstructured state plus typed questions and returns structured values/probability distributions without text generation. It states that questions in one request are evaluated independently and in parallel, and that adding questions barely changes response time. The primitive API exposes Choice, Score, and Noul; Choice and Score return full probability distributions, and Noul returns a probability-like scalar. TypeSafe says System One models are trained for calibrated decisions and that probabilities are optimized against outcomes to reflect uncertainty. Its confidence documentation further clarifies that Choice/Score `confidence` is computed from the returned probability distribution rather than being an independent prediction.

TypeSafe's launch post additionally claims a new architecture, a parallel sampler, and a training method named **Reinforcement Learning for Calibrated Decisions (RLCD)**. The company describes sampling as parallel rather than token-sequential, gives 70–500 ms end-to-end latency as the intended operating range, and states that high-cardinality Choice supports up to 255 options. The same post explicitly says that higher-cardinality Choice uses a two-stage system: independent scoring followed by an explicit choice.

These are product/company claims; no public reproducible RLCD paper or full architecture specification was located as of this study date.

---

## 3. Experimental material and reproducibility

Five recorded result directories were used:

- `../jev-results/run_20260919T201050Z_core` — 303-request concurrent architecture/robustness benchmark.
- `../jev-results/run_20260919T201739Z_core` — 26-request isolated `concurrency=1` architecture probe.
- `../jev-results/run_20260919T202139Z_full` — 31-request isolated full scaling probe reaching 128 questions, 255 choices, and approximately 21K input tokens.
- `../jev-results/run_20260919T203758Z_calibration_full` — 5,810-request synthetic exact-probability study.
- `../jev-results/run_20260919T211437Z_semantic_full` — 1,110-request six-domain semantic study.

Each directory is self-contained and preserves its own `manifest.json`, generated `cases.jsonl`, raw API `raw.jsonl`, normalized data, summary, report, and figures. No legacy source-result ZIP archives are required by the analysis.

All API request/response bodies were preserved in JSONL together with token usage, wall-clock latency, timestamps, expected targets, and response headers. The response header `x-envoy-upstream-service-time` was used as the preferred server-path latency measure for isolated architecture tests because client wall time includes network/TLS/proxy overhead.

### Statistical methods

- Binary exact-probability recovery: absolute error, signed bias, RMSE, Pearson correlation, Brier/NLL where sampled outcomes exist.
- Distribution recovery: total variation (TV), Jensen–Shannon divergence (JSD), target-to-prediction KL, soft Brier, maximum-probability and entropy gaps.
- Accuracy proportions: Wilson 95% intervals.
- Mean error/gap estimates: nonparametric bootstrap 95% percentile intervals (10,000 resamples, seed 260919).
- Selective prediction: error rate after retaining the highest-confidence fraction of predictions.
- Architecture timing: isolated concurrency=1 runs; reported upstream service time rather than client latency.

No multiple-hypothesis-correction claim is made. The architecture probes are designed primarily around large effect sizes and scaling shape rather than small p-values.

---

## 4. Architecture and serving behavior

### 4.1 Parallel question scaling

The full isolated run produced:

''' + '\n'.join(q_lines) + rf'''

The endpoint ratio is the key observation. Nominal output accounting rises from {int(q1.output_tokens)} to {int(qn.output_tokens)} tokens (**{qn.output_tokens/q1.output_tokens:.1f}×**), while upstream time changes from {int(q1.upstream_ms)} to {int(qn.upstream_ms)} ms (**{qn.upstream_ms/q1.upstream_ms:.2f}×**). Spearman correlation between nominal output-token count and upstream latency across these eight points is {q_rho:.3f}; the curve is noisy rather than proportional.

**Observed:** all questions returned real answers; the additional questions were not ignored.  
**Supported inference:** the reported output-token count cannot be interpreted as ordinary sequentially decoded output tokens. A conventional autoregressive decoder would incur serial work proportional to generated sequence length; the observed 106.8× increase in nominal output accounting without comparable latency growth is incompatible with that interpretation.  
**Consistent with public claim:** TypeSafe states that questions are evaluated in parallel and that adding questions barely changes response time.

![Parallel question scaling](figures/question_scaling.png)

### 4.2 Choice cardinality scaling

''' + '\n'.join(c_lines) + rf'''

From 2 to 255 choices, nominal output accounting rises **{cn.output_tokens/c1.output_tokens:.1f}×**, while measured upstream time is {int(c1.upstream_ms)} ms at 2 choices and {int(cn.upstream_ms)} ms at 255 choices. Spearman correlation between nominal output accounting and latency is {c_rho:.3f}; again the relationship is not remotely compatible with sequential generation of thousands of response tokens.

TypeSafe publicly states that high-cardinality Choice uses a two-stage mechanism of independent scoring followed by explicit choice. The black-box data are **consistent** with a vectorized first-stage scorer. No reproducible latency discontinuity uniquely identifies the internal transition, so the exact two-stage implementation remains unobserved.

![Choice cardinality scaling](figures/choice_scaling.png)

### 4.3 Context-length scaling and evidence position

The full isolated context probe produced:

''' + '\n'.join(ctx_lines) + rf'''

Unlike question/candidate count, context length eventually increases service time: approximately 21K-token states generally require more time than the ~0.5–7K-token cases. This separation is architecturally suggestive: **state encoding cost grows with state length, while decision fan-out cost is comparatively cheap and parallelized.** Evidence remained correctly usable when placed at the beginning, middle, or end in the tested contexts; no stable lost-in-the-middle failure was demonstrated.

![Context scaling](figures/context_scaling.png)

### 4.4 Runtime-defined semantics and invariances

The core benchmark tested candidate-order permutations, opaque candidate keys, question paraphrases, question bundling/independence, irrelevant context, evidence removal, contradiction, prompt-injection-like text inside state, repeatability, and cross-domain cases. Jev correctly interpreted semantically defined runtime candidates even when option identifiers were randomized, supporting the conclusion that Choice is not a fixed-label classifier. The model also reacted appropriately to removal of decisive evidence rather than blindly retaining the original class.

Probabilities exposed by the API were quantized to two decimal places in these runs, while Score/confidence behavior indicated that higher-precision internal values likely exist before presentation rounding. Small response variation across identical requests further indicates stochastic inference, backend/numeric variability, ensembling, or some combination; the API alone cannot identify which.

---

## 5. Exact probabilistic calibration

### 5.1 Overall binary probability recovery

Across {cal_global['n']:,} binary rows with analytically known target probabilities:

- MAE: **{cal_global['mae'][0]:.4f}** (95% bootstrap CI {cal_global['mae'][1]:.4f}–{cal_global['mae'][2]:.4f}).
- RMSE: **{cal_global['rmse']:.4f}**.
- signed bias: **{cal_global['bias'][0]:+.4f}** (95% CI {cal_global['bias'][1]:+.4f}–{cal_global['bias'][2]:+.4f}).
- Pearson correlation with exact target probability: **{cal_global['corr']:.3f}**.

The high correlation means Jev generally moves probability in the correct direction. The MAE/RMSE show that this should not be confused with exact posterior recovery.

![Exact probability recovery](figures/exact_probability_recovery.png)

### 5.2 Error by task family

''' + '\n'.join(cal_lines) + rf'''

Direct aleatoric probability is substantially easier for Jev than deriving posteriors from conditional evidence. Rare-base-rate problems show the largest systematic difficulty among the main Bayesian families. Equivalent mathematical problems rendered in different surface forms also produce materially different outputs.

For the representation-invariance groups, the mean within-problem range across equivalent renderings is **{rep_range_stats[0]:.4f}** (95% bootstrap CI {rep_range_stats[1]:.4f}–{rep_range_stats[2]:.4f}); the maximum observed range is **{rep_range_max:.4f}**. This is evidence of surface-representation dependence, not merely random sampling noise.

![Calibration error by experiment](figures/calibration_mae.png)

### 5.3 Multiclass Choice posterior recovery

On {multi_stats['n']:,} three-class Bayesian cases:

- hard top-class accuracy: **{pct(multi_stats['accuracy'])}**;
- mean TV distance: **{multi_stats['tv'][0]:.3f}** (95% bootstrap CI {multi_stats['tv'][1]:.3f}–{multi_stats['tv'][2]:.3f});
- mean JSD: **{multi_stats['jsd'][0]:.3f}** (95% CI {multi_stats['jsd'][1]:.3f}–{multi_stats['jsd'][2]:.3f});
- exact posterior mean max probability: **{multi_stats['target_max'][0]:.3f}**;
- Jev mean max probability: **{multi_stats['pred_max'][0]:.3f}**;
- paired mean sharpness/overconfidence gap: **{multi_stats['overgap'][0]:+.3f}** (95% CI {multi_stats['overgap'][1]:+.3f}–{multi_stats['overgap'][2]:+.3f});
- exact posterior entropy: **{multi_stats['target_entropy'][0]:.3f} nats** vs Jev **{multi_stats['pred_entropy'][0]:.3f} nats**.

This is a large effect: Choice tends to collapse distributions toward a dominant class much more strongly than the supplied data-generating process warrants. The effect is systematic and survives a large sample.

### 5.4 Interpretation of the calibration result

This experiment does **not** prove that RLCD fails on its intended training distribution. It establishes a narrower result: when the state explicitly supplies a known probabilistic generative model, Jev's outputs do not behave as exact Bayesian posteriors, and multiclass Choice is substantially sharper than the mathematically correct posterior distribution.

That distinction matters because TypeSafe describes calibration as a population property. A decision model can rank uncertainty usefully while still being miscalibrated on a shifted task family. This is precisely why calibration literature emphasizes distribution-specific evaluation and why post-hoc calibration methods are routinely needed for neural classifiers.

---

## 6. Semantic calibration

### 6.1 Overall semantic result

The full semantic suite contains {sem_stats['n']:,} successful Choice decisions spanning incident triage, hardware fault attribution, security classification, support routing, code-review severity, and policy/compliance decisions.

- modal accuracy: **{pct(sem_stats['accuracy'])}** (Wilson 95% CI {pct(sem_stats['acc_ci'][0])}–{pct(sem_stats['acc_ci'][1])});
- mean TV to reference distribution: **{sem_stats['tv'][0]:.3f}** (95% bootstrap CI {sem_stats['tv'][1]:.3f}–{sem_stats['tv'][2]:.3f});
- mean JSD: **{sem_stats['jsd'][0]:.3f}** (95% CI {sem_stats['jsd'][1]:.3f}–{sem_stats['jsd'][2]:.3f});
- mean reference modal probability: **{sem_stats['refmax'][0]:.3f}**;
- mean Jev top probability: **{sem_stats['jevmax'][0]:.3f}**;
- paired mean overconfidence gap: **{sem_stats['overgap'][0]:+.3f}** (95% CI {sem_stats['overgap'][1]:+.3f}–{sem_stats['overgap'][2]:+.3f});
- mean reference entropy: **{sem_stats['refentropy'][0]:.3f} nats**; Jev entropy: **{sem_stats['predentropy'][0]:.3f} nats**.

The overconfidence gap CI excludes zero by a very wide margin. Under this benchmark's authored adjudication distributions, Jev is consistently much sharper than the stated ambiguity.

![Semantic confidence](figures/semantic_confidence.png)

### 6.2 Ambiguity strata

''' + '\n'.join(ambig_lines) + rf'''

The critical pattern is not merely lower accuracy on hard cases. Jev frequently identifies the same modal class as the reference while assigning much more probability mass to it. In other words, top-1 semantic judgment is often strong even when distribution matching is poor.

### 6.3 Domain breakdown

''' + '\n'.join(domain_lines) + rf'''

Performance is therefore domain-dependent. Security classification and incident triage are especially strong on modal accuracy; support routing and hardware attribution are materially weaker. This is further evidence against treating one global probability threshold as universally meaningful.

![Domain overconfidence](figures/semantic_domain_overconfidence.png)

### 6.4 Robustness to irrelevant context

For the semantic-adjudication subset:

''' + '\n'.join(noise_lines) + rf'''

No monotonic degradation with irrelevant-context injection is visible at these tested noise levels. This supports strong semantic robustness in the evaluated range.

### 6.5 Confidence as a selective-risk signal

Sorting predictions by returned confidence/top probability and retaining only the most confident subset gives:

''' + '\n'.join(risk_lines) + rf'''

This is one of Jev's strongest practical results. Even though the absolute probability values are sharper than the reference distributions, confidence still provides a useful **ranking** for automation/escalation. That is the selective-classification use case: abstain/escalate as uncertainty rises rather than assume every score is an externally calibrated posterior.

![Semantic risk coverage](figures/semantic_risk_coverage.png)

### 6.6 Limitation of the semantic reference

The semantic reference distributions are deliberately transparent, authored “20-vote-style” adjudication distributions rather than votes collected from independent human experts. Therefore, the +{sem_stats['overgap'][0]:.3f} semantic gap is not an objective proof that Jev is overconfident by exactly that amount in the real world. It **does** prove that Jev is much sharper than the explicit ambiguity structure encoded by this benchmark. A publication-quality semantic-calibration claim would require independently collected multi-annotator labels, ideally with adjudicator expertise appropriate to each domain.

---

## 7. Reverse-engineered architecture

### 7.1 Constraints imposed by the observations

Any candidate architecture must jointly explain:

- typed outputs without free-form decoding;
- arbitrary runtime-defined questions and candidate descriptions;
- 128 questions in one request with only modest latency growth;
- 255-choice decisions without latency proportional to nominal output length;
- rising cost with large state/context length;
- question independence within a request;
- high-cardinality Choice behavior consistent with TypeSafe's disclosed independent-scoring/two-stage description;
- full probability distributions and a confidence statistic derived from those distributions.

### 7.2 Best-supported structure

The most parsimonious reconstruction is:

```text
UNSTRUCTURED STATE
      |
      v
+---------------------------+
| shared semantic backbone  |
| / state representation    |
+-------------+-------------+
              |
      +-------+-------------------------------+
      |                                       |
      v                                       v
runtime question representations      runtime candidate/level representations
      |                                       |
      +----------------+----------------------+
                       |
                       v
             vectorized interaction/scoring
                       |
          +------------+-------------+
          |            |             |
          v            v             v
        Noul         Choice         Score
      Bernoulli    categorical     ordinal/
       score       distribution    level distribution
```

A mathematical abstraction is:

$$
H_x = E_\theta(x)
$$

$$
z_{{ij}}=g_\theta(H_x, q_i, c_{{ij}})
$$

$$
P(c_{{ij}}\mid x,q_i)=\mathrm{{softmax}}_j(z_{{ij}})
$$

where the state representation is shared and many question/candidate interactions are executed as batched tensor operations. For Noul, the candidate space is effectively binary; Score likely maps ordered levels to a distribution and computes a scalar position from that distribution.

This is a **functional reconstruction**, not a claim that TypeSafe uses exactly these modules or equations.

### 7.3 Relative plausibility of architecture families

Our posterior judgment after the experiments is approximately:

- **~70%:** shared semantic encoder/backbone + dynamic question/candidate representations + vectorized decision scorers/typed heads.
- **~18%:** masked/latent answer-slot Transformer where typed answer positions are resolved in parallel.
- **~8%:** causal Transformer retained primarily as a representation backbone, with autoregressive decoding bypassed for Jev outputs.
- **~4%:** substantially different architecture not captured above.

These percentages are subjective hypothesis weights, not statistical posterior probabilities from a formal generative model.

### 7.4 What the black box cannot identify

The experiments do **not** reveal:

- model parameter count;
- encoder-only vs hybrid attention masks with certainty;
- dense vs sparse/MoE execution;
- whether question/candidate embeddings are cached or jointly cross-attended;
- the exact high-cardinality shortlist/reranker transition;
- the exact confidence formula;
- RLCD's reward, optimization algorithm, calibration regularizer, teacher data, or training corpus;
- whether training uses distillation from frontier LLMs.

Any stronger claim would exceed the evidence.

---

## 8. What the results imply about RLCD

TypeSafe frames RLCD as optimizing “epistemically honest probabilities.” Proper-scoring-rule theory provides the standard mathematical mechanism for incentivizing truthful probabilistic forecasts in expectation; calibration theory separately distinguishes accuracy, sharpness/resolution, and reliability. Without a public RLCD specification, the API results cannot establish which of these objectives TypeSafe actually optimizes.

The observed behavior suggests that RLCD, or the overall training stack, produces at least three desirable properties:

1. **strong semantic decision quality** on many decision-shaped tasks;
2. **useful ordering of confidence** for selective automation;
3. **stable, structured probability outputs** rather than prose-level self-reported confidence.

It does **not** establish universal posterior calibration. Exact Bayesian tests show representation-sensitive error and multiclass over-sharpening; semantic tests show distributions substantially sharper than the benchmark's ambiguity reference. A defensible operational interpretation is therefore:

> Jev's probabilities are useful model-native decision scores whose calibration should be validated on the target task distribution before they are treated as literal probabilities for irreversible automation.

This interpretation is also compatible with TypeSafe's own documentation, which says threshold values depend on the domain and recommends testing on one's own data.

---

## 9. Jev versus a conventional LLM

The fundamental distinction demonstrated here is not simply “small model versus large model.” It is the computational contract.

A conventional causal LLM models a token sequence:

$$
P(t_1,\ldots,t_T\mid x)=\prod_{{k=1}}^T P(t_k\mid x,t_{{\lt k}})
$$

Even when grammar-constrained, structured output is ordinarily serialized through sequential token decoding.

The Jev observations are instead consistent with direct evaluation of bounded semantic answer spaces:

$$
(\text{state}, \text{question}, \text{candidates}) \rightarrow \Delta^K
$$

where $\Delta^K$ is a probability simplex over developer-provided candidates. Giving up unrestricted string generation permits much greater batching and removes the inherently serial output loop. This readily explains why thousands of nominal output-accounting tokens need not correspond to thousands of decoder steps.

The trade-off is equally important: Jev cannot replace a generative model for code generation, explanation, arbitrary synthesis, or long deliberative reasoning. Its most natural role is inside software as a fast semantic judgment layer, optionally escalating uncertain cases to a larger reasoning model or human.

---

## 10. Comparison with the 2025 SalesRLAgent / “I built Jev” claim

The earlier SalesRLAgent work shares the broad philosophy of mapping semantic state to a bounded probability/confidence value and using that result in software rather than generating prose. That makes it a legitimate conceptual predecessor to the System-One-style product pattern.

However, Jev's observed capability is materially more general: runtime questions and candidate sets define new decision functions without retraining a task-specific output head. Functionally, the distinction is approximately:

$$
f_{{sales}}(x)\rightarrow P(\text{{conversion}})
$$

versus

$$
f(x,q,C)\rightarrow P(C\mid x,q).
$$

Thus “built a domain-specific predecessor embodying similar principles” is supported; “implemented the same general Jev architecture” is not established by the 2025 paper alone.

---

## 11. Scientific strength of the evidence

### High-confidence conclusions

1. **Jev is not behaving like ordinary long autoregressive structured-output decoding.** The timing/output-scaling mismatch is too large.
2. **Questions and candidates are processed with substantial parallel/vectorized computation.** This directly reproduces TypeSafe's published parallelism claim.
3. **Choice is runtime-semantic rather than fixed-label.** Opaque keys plus semantic descriptions remain usable.
4. **The confidence signal is operationally useful for ranking semantic risk.** Selective-risk curves improve materially at reduced coverage.
5. **Universal posterior calibration is not supported.** Large exact-probability experiments reveal systematic errors and distribution-shape distortions.

### Medium-confidence conclusions

1. Shared state encoding followed by vectorized decision scoring is the most economical architecture that explains the observations.
2. Choice and Noul likely have materially different output heads/objectives/calibration behavior, given the observed multiclass sharpening versus binary compression patterns.
3. The model contains a higher-precision internal distribution than the two-decimal API representation, inferred from Score/confidence inconsistencies with rounded probabilities.

### Unproven

1. Exact model architecture and parameter count.
2. RLCD's specific optimization mathematics.
3. “Frontier intelligence” across arbitrary task distributions.
4. Calibration on unseen real production distributions.
5. Whether the pricing reflects sustainable underlying inference cost.

---

## 12. Practical deployment implication

The data favor a deployment architecture in which Jev is treated as a **fast decision/routing layer**, but raw probabilities are calibrated and validated per use case:

```text
state
  |
  v
Jev typed decisions
  |
  v
application-specific calibration / threshold policy
  |
  +--> high-confidence, low-risk --> deterministic action
  |
  +--> medium-confidence ---------> verification / larger model
  |
  +--> low-confidence/high-risk --> human or deliberate reasoning model
```

For high-stakes irreversible actions, empirical risk-versus-coverage on production-like data should determine thresholds. A hard-coded global rule such as `if confidence > 0.95: execute()` is not justified by these experiments.

---

## 13. Limitations and threats to validity

- All observations concern `jev-1.13.0` as served on 19 September 2026; TypeSafe may update `jev-latest`.
- API service time is an Envoy-reported upstream duration, not GPU kernel time. Queueing, batching, and replica selection may contribute.
- Architecture scaling has few repetitions at each extreme point; the effects are nevertheless orders of magnitude larger than timing noise for the autoregressive-vs-parallel question.
- Synthetic Bayes tasks test probabilistic reasoning under explicit generative models; they may differ from RLCD's training distribution.
- Semantic reference distributions are authored adjudication priors rather than independent human-panel frequencies.
- No comparison model was run through exactly the same local harness in this study; conclusions about Jev are primarily absolute/structural, not a full model leaderboard.
- No access to TypeSafe weights, training data, server traces, or architecture source was available.

---

## 14. Recommended next experiments

The highest-value next study is **human-panel semantic calibration**, not more repetitions of the authored semantic set. A rigorous follow-up would collect 5–10 independent qualified judgments per case across 200–500 ambiguous operational cases, reserve a held-out test set, and evaluate Jev's full distributions against empirical vote frequencies and adjudicated outcomes.

A second valuable comparison is to run a strong open baseline through an equivalent non-generative decision head—for example a modern bidirectional encoder and a small Qwen/Gemma backbone with dynamic candidate scoring—to isolate how much of Jev's advantage comes from architecture versus training/data.

Finally, for production use, fit a task-specific post-hoc calibrator (temperature/vector/Dirichlet or isotonic depending on primitive and sample size) on held-out real decisions, then evaluate selective risk at action-specific thresholds.

---

## 15. Conclusion

The experiments support a substantive technical result behind the TypeSafe story. Jev behaves like a **decision-native, massively parallel semantic model**, not like a conventional LLM merely forced to emit JSON. Its runtime-defined typed interface, flat fan-out scaling, and approximately 100–200 ms upstream behavior make the architectural idea credible and practically interesting.

The experiments do **not** support treating every returned number as a universally calibrated Bayesian probability. Jev is highly competent at direct probability reading and semantic classification, but harder posterior derivation, rare base rates, equivalent numerical representations, and multiclass ambiguity expose systematic distortion—especially sharp Choice distributions. On semantic decisions, the confidence signal nevertheless ranks error well enough to be useful for selective automation.

The most evidence-consistent interpretation is therefore:

> **Jev is a fast, general, runtime-programmable probabilistic decision model with strong semantic ranking and task-dependent calibration. Its architectural claims are substantially supported by black-box measurements; its broad “epistemically honest probability” claim requires distribution-specific validation and should not be interpreted as universal posterior calibration.**

That conclusion is more modest than the strongest marketing language, but it still describes a genuinely useful and technically differentiated model class.

---

## References

1. TypeSafe AI. **Introduction.** https://docs.typesafe.ai/introduction (accessed 2026-09-19).
2. TypeSafe AI. **System One.** https://docs.typesafe.ai/concepts/system-one (accessed 2026-09-19).
3. TypeSafe AI. **Primitives (Questions).** https://docs.typesafe.ai/primitives (accessed 2026-09-19).
4. TypeSafe AI. **Confidence.** https://docs.typesafe.ai/confidence (accessed 2026-09-19).
5. Almeida, D. **Introducing System One Models & Jev.** TypeSafe AI, 15 Sep 2026. https://typesafe.ai/blog/introducing-system-one-models-and-jev
6. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. **On Calibration of Modern Neural Networks.** ICML 2017. arXiv:1706.04599. https://arxiv.org/abs/1706.04599
7. Silva Filho, T. et al. **Classifier Calibration: A Survey on How to Assess and Improve Predicted Class Probabilities.** Machine Learning (survey preprint arXiv:2112.10327). https://arxiv.org/abs/2112.10327
8. Geifman, Y., El-Yaniv, R. **Selective Classification for Deep Neural Networks.** NeurIPS 2017. arXiv:1705.08500. https://arxiv.org/abs/1705.08500
9. Fisch, A., Jaakkola, T., Barzilay, R. **Calibrated Selective Classification.** 2022. arXiv:2208.12084. https://arxiv.org/abs/2208.12084
10. Waghmare, K., Ziegel, J. **Proper Scoring Rules for Estimation and Forecast Evaluation.** Annual Review of Statistics and Its Application 13 (2026): 271–296. https://doi.org/10.1146/annurev-statistics-042424-050626
11. Gneiting, T., Raftery, A. E. **Strictly Proper Scoring Rules, Prediction, and Estimation.** JASA 102 (2007): 359–378.
12. TypeSafe benchmark data produced by the black-box harness accompanying this report; the five self-contained recorded run directories are listed in Section 3.

---

## Appendix A. Exact study-run reproduction commands

All commands below are run from the bundled `benchmark/` directory after installation and after setting `TYPESAFE_API_KEY`.

Architecture/robustness core, concurrent (303 requests):

```bash
jevbench --config config/architecture.yaml run --profile core --concurrency 4
```

Architecture core, isolated scaling (26 requests):

```bash
jevbench --config config/architecture.yaml run \
  --profile core --concurrency 1 \
  --only parallel_question_scaling \
  --only choice_cardinality_scaling \
  --only context_length_position
```

Architecture full, isolated boundary scaling (31 requests):

```bash
jevbench --config config/architecture.yaml run \
  --profile full --concurrency 1 \
  --only parallel_question_scaling \
  --only choice_cardinality_scaling \
  --only context_length_position
```

Exact probabilistic calibration, full (5,810 requests):

```bash
jevbench --config config/calibration.yaml run --profile calibration_full --concurrency 4
```

Semantic calibration, full (1,110 requests):

```bash
jevbench --config config/semantic.yaml run --profile semantic_full --concurrency 4
```

For architecture latency inference, use `x-envoy-upstream-service-time` from `raw.jsonl`. Preserve raw JSONL, manifests, token accounting, response headers, and returned distributions for auditability.

## Appendix B. Interpretation guide

- **Probability calibration** asks whether predictions assigned probability $p$ occur at frequency $p$ on a defined population.
- **Sharpness** measures concentration of the predictive distribution; sharpness without calibration can be dangerous.
- **Selective risk** asks how error changes as low-confidence cases are rejected/escalated.
- **TV/JSD/KL** measure distance between full distributions, not merely agreement on the top class.
- A model may have high top-1 accuracy and still poorly reproduce uncertainty; Jev demonstrates exactly why these metrics must be reported separately.

## Appendix C. Accompanying artifact bundle

The complete bundle contains the unified `benchmark/` codebase, this `report/`, and all five self-contained raw `../jev-results/` directories. Older development benchmark packages and duplicate source-result archives are intentionally excluded because `benchmark/` v1.0.0 and the extracted result directories supersede them.
'''

(OUT/'JEV_SYSTEM_ONE_BLACKBOX_REPORT.md').write_text(report)
# machine-readable stats
stats={'architecture':{'question_scaling':q.to_dict(orient='records'),'choice_scaling':c.to_dict(orient='records'),'context_scaling':ctx.to_dict(orient='records')},'calibration':{'global':cal_global,'by_experiment':cal_stats,'multiclass':multi_stats,'representation_range':rep_range_stats,'representation_max_range':rep_range_max},'semantic':{'overall':sem_stats,'ambiguity':ambig_stats,'by_domain':by_domain,'noise':noise,'risk_coverage':risk}}
# sanitize numpy
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,)):return None if np.isnan(x) else float(x)
    if isinstance(x,float) and np.isnan(x):return None
    return x
(OUT/'jev_computed_statistics.json').write_text(json.dumps(clean(stats),indent=2))
print(OUT/'JEV_SYSTEM_ONE_BLACKBOX_REPORT.md')
print(OUT/'jev_computed_statistics.json')
