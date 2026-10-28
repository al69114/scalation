"""Recompute final PyTorch fits using submitted definitions and saved selections.
Run from repository root: MPLCONFIGDIR=/tmp/project3-mpl python3 project3/audit/verify_results.py
Does not rerun full hyperparameter searches or overwrite original results.
"""
import ast, json, pathlib, contextlib, io
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
ROOT = pathlib.Path(__file__).resolve().parents[2]
torch.set_num_threads(1)

def definitions(label):
    path = ROOT / 'project3' / f'NeuralNet_{label}.ipynb'
    if label == '5L':
        path = ROOT / 'project3/NeuralNet_5L.py'
        source = path.read_text()
    else:
        nb = json.loads(path.read_text())
        source = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and not ''.join(c['source']).startswith('%'))
    tree = ast.parse(source)
    keep=[]
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef)):
            keep.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='activations' for t in node.targets):
            keep.append(node)
    ns={'__file__':str(path)}
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(path),'exec'),ns)
    return ns

rows=[]
for label in ['2L','3L','4L','5L','Custom']:
    ns=definitions(label)
    for name,stem,target in [('Airfoil','airfoil','scaled_sound_pressure_db'),('Auto MPG','auto_mpg','mpg'),('Concrete','concrete','compressive_strength_mpa')]:
        df=pd.read_csv(ROOT/f'data/{stem}.csv')
        X=df.drop(columns=target); y=df[target]
        if label=='5L': X=X.to_numpy(dtype=np.float32); y=y.to_numpy(dtype=np.float32)
        xt,xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42)
        xs=StandardScaler(); xt=xs.fit_transform(xt); xv=xs.transform(xv)
        ys=StandardScaler(); yt_s=ys.fit_transform(np.asarray(yt).reshape(-1,1))
        if label=='2L':
            p={'learning_rate':.3,'batch_size':64 if name=='Auto MPG' else 16,'best_epoch':{'Airfoil':41,'Auto MPG':45,'Concrete':38}[name]}
        elif label=='5L':
            p=pd.read_csv(ROOT/'project3/outputs/5L_results.csv').set_index('Dataset').loc[name].to_dict()
        else:
            tuning=pd.read_csv(ROOT/f'project3/results/{label}_tuning_{stem}.csv')
            p=tuning.sort_values('val_mse').iloc[0].to_dict()
            for key in ['hidden_size','hidden1','hidden2','width','batch_size','best_epoch']: 
                if key in p: p[key]=int(p[key])
            print(label,name,'grid rows',len(tuning),'best',p,flush=True)
        if label=='5L':
            model,_,_=ns['train_model'](xt,yt_s.ravel(),ast.literal_eval(p['Hidden Layers']),p['Activation'],p['Learning Rate'],int(p['Batch Size']),epochs=int(p['Best Epoch']))
            pred=lambda a:ys.inverse_transform(ns['predict'](model,a).reshape(-1,1)).ravel()
        else:
            t=torch.tensor(xt,dtype=torch.float32)
            u=torch.tensor(np.asarray(yt).reshape(-1,1) if label=='2L' else yt_s,dtype=torch.float32)
            model=ns['train_final_model'](t,u,p)
            def pred(a):
                model.eval()
                with torch.no_grad(): v=model(torch.tensor(a,dtype=torch.float32)).numpy()
                return v.ravel() if label=='2L' else ys.inverse_transform(v).ravel()
        row={'Model':label,'Dataset':name}
        for split,a,b in [('Train',xt,yt),('Validation',xv,yv)]:
            yp=pred(a); mse=mean_squared_error(b,yp)
            row.update({f'{split} R²':r2_score(b,yp),f'{split} MSE':mse,f'{split} RMSE':np.sqrt(mse)})
        if label in ['3L','4L','Custom']:
            expected=pd.read_csv(ROOT/f'project3/results/{label}_qof.csv').set_index('Dataset').loc[name]
            row['Max absolute difference from saved metrics']=max(abs(row[k]-expected[k]) for k in expected.index)
        print(row,flush=True); rows.append(row)
        pd.DataFrame(rows).to_csv(ROOT/'project3/audit/recomputed_pytorch.csv',index=False)
print('Final-fit verification complete. No full grids rerun.')
