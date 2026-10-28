"""Audit adaptation: expose the 4L notebook's fixed training seed as a parameter.
This supplementary check does not change the submitted notebook.
Run from repository root with MPLCONFIGDIR=/tmp/project3-mpl.
"""
import ast,json,pathlib
import numpy as np,pandas as pd,torch
from sklearn.model_selection import train_test_split
ROOT=pathlib.Path(__file__).resolve().parents[2]
nb=json.loads((ROOT/'project3/NeuralNet_4L.ipynb').read_text())
tree=ast.parse('\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'))
keep=[]
for node in tree.body:
    if isinstance(node,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.FunctionDef)):
        if isinstance(node,ast.FunctionDef) and node.name=='train_final_model':
            node.args.args.append(ast.arg(arg='seed')); node.args.defaults.append(ast.Constant(42))
            for call in ast.walk(node):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='manual_seed':call.args=[ast.Name(id='seed',ctx=ast.Load())]
        keep.append(node)
    elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='activations' for t in node.targets):keep.append(node)
ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=keep,type_ignores=[])),'4L_seed_audit','exec'),ns)
torch.set_num_threads(1)
rows=[]
for name,stem,target in [('Airfoil','airfoil','scaled_sound_pressure_db'),('Auto MPG','auto_mpg','mpg'),('Concrete','concrete','compressive_strength_mpa')]:
    df=pd.read_csv(ROOT/f'data/{stem}.csv')
    split=train_test_split(df.drop(columns=target),df[target],test_size=.2,random_state=42)
    t,ys=ns['prepare_data'](*split)
    p=pd.read_csv(ROOT/f'project3/results/4L_tuning_{stem}.csv').sort_values('val_mse').iloc[0].to_dict()
    for k in ['hidden1','hidden2','batch_size','best_epoch']:p[k]=int(p[k])
    r=[]
    for seed in range(5):
        m=ns['train_final_model'](t['X_train'],t['y_train_s'],p,seed)
        r.append(ns['evaluate_model'](m,t['X_test'],t['y_test'],ys)[0])
    row={'Dataset':name,'Mean Validation R²':np.mean(r),'SD':np.std(r),'Min':min(r),'Max':max(r),'Seeds 0-4':str(r)}
    rows.append(row);print(row,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'project3/audit/recomputed_4L_seeds.csv',index=False)
