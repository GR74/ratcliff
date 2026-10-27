"""Bounded, one-flag padding fidelity/performance screen on frozen Model B."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from model_b_bads_support import write_json


def main(args):
    root, out, report = [Path(v).resolve() for v in [args.baseline, args.output, args.report]]
    out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    if (out/'status.json').exists():
        raise RuntimeError('Preserve previous output; do not overwrite an experiment')
    source = root/'source-322bd1f'
    source_manifest = json.loads((root/'source-manifest.json').read_text())
    for relative, expected in source_manifest['files'].items():
        if hashlib.sha256((source/relative.replace('\\','/')).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source hash mismatch: {relative}')
    active = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi', '--query-compute-apps=pid,process_name',
                             '--format=csv,noheader'], capture_output=True, text=True, check=True)
    while args.wait_for_idle and active.stdout.strip():
        state=dict(stage='waiting_for_idle_gpu',pid=os.getpid(),
                   utc=datetime.now(timezone.utc).isoformat(),other_compute_processes=active.stdout.strip())
        write_json(out/'status.json',state)
        write_json(report/'evidence.json',dict(status=state,results={'decision':'pending'}))
        (report/'README.md').write_text('# RTX 4060: smaller K-L padding screen\n\n'
            '**Status: waiting for another GPU workload to finish.**\n\n'
            'No candidate simulations or timings have started. The 20-minute screen budget starts after the GPU is free.\n')
        time.sleep(30)
        active=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name',
            '--format=csv,noheader'],capture_output=True,text=True,check=True)
    if active.stdout.strip():
        raise RuntimeError('GPU already in use: '+active.stdout.strip())
    started = time.perf_counter()
    results = {'decision': 'pending', 'basis_checks': [], 'path_checks': [], 'timing_pairs': []}
    def save(stage, **extra):
        elapsed = time.perf_counter()-started
        status = dict(stage=stage, pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=elapsed, **extra)
        write_json(out/'status.json', status)
        write_json(report/'evidence.json', dict(status=status, results=results))
        lines = ['# RTX 4060: smaller K-L padding screen', '', f'**Status: {stage}.**', '',
            'Only `kl_bucketed=True` changes. Retained modes, noise indexing and model definition are unchanged.', '',
            f"Decision: **{results['decision']}**. Elapsed: {elapsed/60:.2f} minutes.", '',
            f"Basis checks: {len(results['basis_checks'])}; path checks: {len(results['path_checks'])}; objective timing pairs: {len(results['timing_pairs'])}/30.", '']
        if 'timing' in results:
            t=results['timing']
            lines += [f"Control median: {t['control_median']:.3f} s; candidate median: {t['candidate_median']:.3f} s.",
                f"Candidate/control ratio: {t['ratio']:.3f}; descriptive 95% interval {t['interval_95']}.", '']
        if 'failure' in results:
            lines += [results['failure'], '']
        lines += ['No full fit is launched by this screen. An exact-check failure requires separate numerical/distributional investigation; it does not alone establish scientific harm.', '',
            '[Evidence](evidence.json) · [Protocol](../../plans/2026-10-22-model-b-padding-screen.md)', '']
        (report/'README.md').write_text('\n'.join(lines))
        print(stage, extra, flush=True)
        telemetry=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi',
            '--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'], capture_output=True, text=True)
        with (out/'telemetry.csv').open('a') as handle:
            handle.write(telemetry.stdout)
    def deadline():
        if time.perf_counter()-started >= 1200:
            raise TimeoutError('Twenty-minute screen budget reached; retain partial evidence')
    save('preflight')
    try:
        os.environ.setdefault('JAX_PLATFORMS','cuda')
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
        sys.path.insert(0,str(source))
        import jax
        jax.config.update('jax_enable_x64',True)
        jax.config.update('jax_default_matmul_precision','default')
        import jax.numpy as jnp
        from model_b import grf, grf_kl, objective
        from scripts import benchmark_speedups as bench
        from model_b_baseline_observer import load_observer, sample_observed
        assert jax.default_backend()=='gpu' and '4060' in jax.devices()[0].device_kind
        grf_kl._BASIS_CACHE_MAXSIZE=2
        previous=root/'archived-case'
        protocol=json.loads((previous/'protocol.json').read_text())
        fit=json.loads((previous/'fit-baseline-1-early_stop.json').read_text())
        truth=np.asarray(protocol['truth'])
        if hashlib.sha256((previous/'data-baseline.npz').read_bytes()).hexdigest()!=protocol['data_sha256']:
            raise ValueError('Dataset changed')
        settings = lambda bucket: SimpleNamespace(chunk_a=256,chunk_b=8,block_size=32,
            kl_bucketed=bucket,cumulative_kl=False,stable_kl_noise=True)
        def plain(p, seed, bucket, n):
            deadline()
            return bench.simulate_conditions(jax,'b',p,n,seed,'early_stop',settings(bucket))
        def mode_count(sig):
            deadline()
            eigen=np.asarray(grf.calc_LAM(s1=round(float(sig),6),s2=round(float(sig),6)))**2
            ordered=np.sort(eigen.ravel())[::-1]
            return min(2000,int(np.searchsorted(np.cumsum(ordered)/ordered.sum(),.99)+1))
        widths=[.2,8.,10.,10.000001,12.,17.,fit['x0'][5],fit['params'][5]]
        transitions=[]
        for capacity in [512,1024,1536]:
            lo,hi=.2,17.
            if mode_count(lo)>capacity and mode_count(hi)<=capacity:
                for _ in range(25):
                    mid=(lo+hi)/2
                    if mode_count(mid)>capacity: lo=mid
                    else: hi=mid
                pair=[round(lo-2e-6,6),round(hi+2e-6,6)]
                widths.extend(pair);transitions.append(dict(capacity=capacity,widths=pair))
        widths=list(dict.fromkeys(widths))
        metadata=bench.metadata(jax)
        metadata['enclosing_checkout_commit']=metadata.get('commit')
        metadata['commit']=source_manifest['commit']
        write_json(out/'protocol.json',dict(source_commit=source_manifest['commit'],
            source_manifest_sha256=hashlib.sha256((root/'source-manifest.json').read_bytes()).hexdigest(),
            data_sha256=protocol['data_sha256'],widths=widths,transitions=transitions,
            tool_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
            candidate='kl_bucketed=True only',budget_seconds=1200,timing_vectors=protocol['timing_vectors'],
            baseline_protocol=protocol,device=jax.devices()[0].device_kind,metadata=metadata,
            fit_sha256=hashlib.sha256((previous/'fit-baseline-1-early_stop.json').read_bytes()).hexdigest()))
        for width in widths:
            deadline()
            reference=grf_kl.calc_kl_basis(width,pad_to_k_max=True,return_mode_ids=True)
            candidate=grf_kl.calc_kl_basis(width,pad_to_k_max=True,return_mode_ids=True,
                                         mode_buckets=grf_kl.DEFAULT_MODE_BUCKETS)
            k=reference[1]
            assert k==candidate[1] and reference[2]==candidate[2]
            np.testing.assert_array_equal(np.asarray(reference[0])[:,:2*k],np.asarray(candidate[0])[:,:2*k])
            np.testing.assert_array_equal(np.asarray(reference[0])[:,2*k:],0)
            np.testing.assert_array_equal(np.asarray(candidate[0])[:,2*k:],0)
            np.testing.assert_array_equal(np.asarray(reference[3])[:k],np.asarray(candidate[3])[:k])
            keys=jax.random.split(jax.random.key(0),2)
            a=grf_kl.mode_normal_blocks(keys,reference[3],32)
            b=grf_kl.mode_normal_blocks(keys,candidate[3],32)
            np.testing.assert_array_equal(np.asarray(a)[:,:,:2*k],np.asarray(b)[:,:,:2*k])
            results['basis_checks'].append(dict(sig=width,retained_modes=k,variance=reference[2],
                reference_capacity=reference[0].shape[1]//2,candidate_capacity=candidate[0].shape[1]//2))
            save('basis_checks',width=width)
        observer=load_observer(source)
        probes=[('truth',truth),('start',np.asarray(fit['x0'])),('simplex_fit',np.asarray(fit['params']))]
        for label,boundary,spread in [('low_boundary',1.,.01),('high_boundary',100.,2.)]:
            p=truth.copy();p[2]=boundary;p[3]=spread;probes.append((label,p))
        for width in widths:
            p=truth.copy();p[5]=width;probes.append((f'width_{width}',p))
        mismatch=False
        for label,p in probes:
            for seed in [0,1]:
                deadline()
                a=sample_observed(observer,p,seed,nsim=11,kl_bucketed=False)
                b=sample_observed(observer,p,seed,nsim=11,kl_bucketed=True)
                if label=='truth' and seed==0:
                    for bucket,captured in [(False,a),(True,b)]:
                        rt,cat,_=plain(p,seed,bucket,11)
                        np.testing.assert_array_equal(rt,captured['rt'])
                        np.testing.assert_array_equal(cat,captured['cat'])
                counts={name:int(np.count_nonzero(a[name]!=b[name])) for name in ['rt','cat','steps','positions','crossed']}
                results['path_checks'].append(dict(probe=label,seed=seed,mismatches=counts,
                    trials=22,max_rt_difference=float(np.max(np.abs(a['rt']-b['rt'])))))
                np.savez_compressed(out/f'paths-{label}-{seed}.npz',**{f'{tag}_{name}':v[name] for tag,v in [('control',a),('candidate',b)] for name in ['rt','cat','steps','positions','crossed']})
                mismatch |= any(counts.values())
                save('path_checks',probe=label,seed=seed)
        if mismatch:
            results['decision']='numerical_fidelity_gap'
            results['failure']='At least one small-probe first-passage output differs. Exact-equivalence fast track stopped; no fit or speed claim.'
            save('completed');write_json(out/'summary.json',results);return
        a=plain(truth,0,False,9000);b=plain(truth,0,True,9000)
        counts={name:int(np.count_nonzero(a[i]!=b[i])) for i,name in enumerate(['rt','cat'])}
        results['production_check']=dict(trials=18000,mismatches=counts)
        np.savez_compressed(out/'production-pair.npz',control_rt=a[0],control_cat=a[1],candidate_rt=b[0],candidate_cat=b[1])
        if any(counts.values()):
            results['decision']='numerical_fidelity_gap'
            results['failure']='Production RT/category outputs differ. Exact-equivalence fast track stopped; no full fit.'
            save('completed');write_json(out/'summary.json',results);return
        data={k:jnp.asarray(v) for k,v in dict(np.load(previous/'data-baseline.npz')).items()}
        def timed_loss(p,bucket):
            deadline();grf_kl._basis_cache.clear();grf_kl._basis_cache_order.clear()
            t=time.perf_counter()
            value=float(jax.block_until_ready(objective.fofs_b_new(jnp.asarray(p),data,jax.random.key(321),
                nsim=9000,chunk_size=8,use_kl=True,time_block_size=32,stop_early=True,compact=True,
                stable_kl_noise=True,kl_bucketed=bucket)))
            if not np.isfinite(value):raise ValueError('Nonfinite objective')
            return dict(seconds=time.perf_counter()-t,loss=value)
        jax.clear_caches();grf_kl._basis_cache.clear();grf_kl._basis_cache_order.clear()
        warmups=[]
        for i in range(3):
            for bucket in ([False,True] if i%2==0 else [True,False]):
                warmups.append(dict(bucketed=bucket,**timed_loss(truth,bucket)))
                save('warming_objective',calls=len(warmups))
        results['warmups']=warmups
        for i,p in enumerate(protocol['timing_vectors']):
            pair={'index':i,'sig':p[5]}
            for bucket in ([False,True] if i%2==0 else [True,False]):
                pair['candidate' if bucket else 'control']=timed_loss(p,bucket)
            results['timing_pairs'].append(pair)
            save('timing_objective',pairs_completed=i+1)
            if pair['candidate']['loss']!=pair['control']['loss']:
                results['decision']='numerical_fidelity_gap'
                results['failure']='A changing-width objective differs. Exact-equivalence fast track stopped; all raw scores retained.'
                save('completed');write_json(out/'summary.json',results);return
        a=np.asarray([p['control']['seconds'] for p in results['timing_pairs']])
        b=np.asarray([p['candidate']['seconds'] for p in results['timing_pairs']])
        indices=np.random.default_rng(2201).integers(0,30,(10000,30))
        interval=np.quantile(np.median(b[indices],axis=1)/np.median(a[indices],axis=1),[.025,.975])
        ratio=float(np.median(b)/np.median(a))
        results['timing']=dict(control_median=float(np.median(a)),candidate_median=float(np.median(b)),
            ratio=ratio,interval_95=interval.tolist(),control_total=float(a.sum()),candidate_total=float(b.sum()))
        results['decision']='promising_for_full_fit' if ratio<=.9 and interval[1]<1 and b.sum()<=a.sum() else 'speed_target_not_met'
        write_json(out/'summary.json',results);save('completed')
    except TimeoutError as exc:
        results['decision']='inconclusive_time_cap';results['failure']=str(exc)
        write_json(out/'summary.json',results);save('completed')
    except BaseException as exc:
        results['decision']='execution_failure';results['failure']=repr(exc)
        write_json(out/'summary.json',results);save('error');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--report',required=True)
    parser.add_argument('--wait-for-idle',action='store_true')
    main(parser.parse_args())
