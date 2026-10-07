"""Independently refit the corrected Scala expressions and check saved test evidence."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'project2/results'


def evaluate(name, data):
    if name == 'one': return np.ones(len(data))
    if name == 'x2^1.0x0^-1.0x4^-1.5':
        return data.iloc[:,2].to_numpy()/(data.iloc[:,0].to_numpy()*data.iloc[:,4].to_numpy()**1.5)
    if name.startswith('log1p('): return np.log1p(data[name[6:-1]].to_numpy())
    if '^' in name:
        column,power=name.rsplit('^',1)
        return data[column].to_numpy()**float(power)
    if name in data: return data[name].to_numpy()
    for column in sorted(data.columns,key=len,reverse=True):
        if name.startswith(column+'_'):
            return data[column].to_numpy()*evaluate(name[len(column)+1:],data)
    raise ValueError(name)


def check_metrics(y,yp,r):
    error=y-yp
    values=[1-error@error/np.sum((y-y.mean())**2),np.sqrt(np.mean(error**2)),np.mean(abs(error))]
    np.testing.assert_allclose(values,[r.r2,r.rmse,r.mae],rtol=1e-9,atol=1e-9)


def symbolic(key):
    d=pd.read_csv(ROOT/f'data/{key}.csv')
    split=pd.read_csv(RESULTS/f'symbolic/{key}_split.csv')
    tr=split.loc[split.split=='train','row_index'].to_numpy()
    te=split.loc[split.split=='test','row_index'].to_numpy()
    assert not set(tr)&set(te)
    assert sorted(np.r_[tr,te])==list(range(len(d)))
    c=pd.read_csv(RESULTS/f'symbolic/{key}_coefficients.csv')
    phi=np.column_stack([evaluate(t,d.iloc[:,:-1]) for t in c.term])
    mu=c.training_mean.to_numpy();sd=c.training_scale.to_numpy()
    np.testing.assert_allclose(phi[tr,1:].mean(0),mu[1:],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(phi[tr,1:].std(0),sd[1:],rtol=1e-10,atol=1e-10)
    z=(phi-mu)/sd
    b=np.linalg.lstsq(z[tr],d.iloc[tr,-1].to_numpy(),rcond=None)[0]
    yp=z[te]@b
    exported=pd.read_csv(RESULTS/f'symbolic/{key}_test_predictions.csv')
    assert list(exported.row_index)==list(te)
    np.testing.assert_allclose(yp,exported.prediction,rtol=1e-8,atol=1e-7)
    np.testing.assert_allclose(z[te]@c.coefficient,exported.prediction,rtol=1e-9,atol=1e-8)
    # Check that the report's 11-significant-digit constants still yield its metrics.
    round11=lambda a: np.array([float(f'{x:.11g}') for x in a])
    printed=((phi-round11(mu))/round11(sd))@round11(c.coefficient.to_numpy())
    np.testing.assert_allclose(printed[te],exported.prediction,rtol=1e-7,atol=1e-5)
    m=pd.read_csv(RESULTS/f'symbolic/{key}_metrics.csv')
    assert (m.test_rows_in_selection==0).all()
    for r in m.itertuples():
        if r.model=='selected':
            idx=tr if r.split=='train' else te
            check_metrics(d.iloc[idx,-1].to_numpy(),z[idx]@c.coefficient,r)
    print(f'PASS symbolic {key}: disjoint split, training scaling, independent refit, complete equation and scores')


def regularized():
    d=pd.read_csv(ROOT/'data/auto_mpg.csv')
    split=pd.read_csv(RESULTS/'regularized/auto_mpg_split.csv')
    tr=split.loc[split.split=='train','row_index'].to_numpy();te=split.loc[split.split=='test','row_index'].to_numpy()
    assert not set(tr)&set(te)
    x=d.iloc[:,:-1].to_numpy();y=d.iloc[:,-1].to_numpy()
    mu=x[tr].mean(0);sd=x[tr].std(0,ddof=1);a=(x[tr]-mu)/sd;b=(x[te]-mu)/sd
    metric=pd.read_csv(RESULTS/'regularized/auto_mpg_metrics.csv')
    predictions=pd.read_csv(RESULTS/'regularized/auto_mpg_test_predictions.csv')
    co=pd.read_csv(RESULTS/'regularized/auto_mpg_coefficients.csv')
    tuning=pd.read_csv(RESULTS/'regularized/auto_mpg_tuning.csv')
    for r in metric.itertuples():
        c=co[co.method==r.method].coefficient.to_numpy()
        np.testing.assert_allclose(c[0],y[tr].mean())
        exported=predictions[predictions.method==r.method]
        np.testing.assert_allclose(c[0]+b@c[1:],exported.prediction,rtol=1e-10,atol=1e-10)
        err=y[te]-exported.prediction.to_numpy()
        np.testing.assert_allclose([np.sqrt(np.mean(err**2)),1-err@err/np.sum((y[te]-y[te].mean())**2),np.mean(abs(err))],[r.test_rmse,r.test_r2,r.test_mae])
        grid=tuning[tuning.method==r.method]
        assert float(grid.loc[grid.cv_rmse.idxmin(),'lambda'])==r[2]
        if r.method=='ridge':
            independent=np.linalg.solve(a.T@a+r[2]*np.eye(a.shape[1]),a.T@(y[tr]-y[tr].mean()))
            np.testing.assert_allclose(independent,c[1:],atol=1e-9)
            foldsize=(len(tr)+4)//5
            for candidate in grid.itertuples():
                sse=0.
                for start in range(0,len(tr),foldsize):
                    val=np.arange(start,min(start+foldsize,len(tr)))
                    fit=np.setdiff1d(np.arange(len(tr)),val)
                    xf=x[tr[fit]];xv=x[tr[val]];yf=y[tr[fit]]
                    fm=xf.mean(0);fs=xf.std(0,ddof=1)
                    af=(xf-fm)/fs;av=(xv-fm)/fs
                    beta=np.linalg.solve(af.T@af+candidate[2]*np.eye(af.shape[1]),af.T@(yf-yf.mean()))
                    sse+=np.sum((y[tr[val]]-(yf.mean()+av@beta))**2)
                np.testing.assert_allclose(np.sqrt(sse/len(tr)),candidate.cv_rmse,rtol=1e-9)
        else:
            independent=Lasso(alpha=r[2]/len(tr),fit_intercept=False,tol=1e-12,max_iter=100000).fit(a,y[tr]-y[tr].mean())
            # ScalaTion ADMM uses looser convergence tolerance; compare prediction accuracy.
            np.testing.assert_allclose(b@independent.coef_,b@c[1:],atol=.03)
        print(f'PASS regularized {r.method}: training intercept/scaler, coefficient predictions, test scores and tuning choice')


if __name__=='__main__':
    for key in ['auto_mpg','concrete','airfoil']:symbolic(key)
    regularized()
