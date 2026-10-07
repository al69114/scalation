"""Refresh the full LaTeX report and figures from the corrected Scala results.
Run the Scala symbolic/regularized mains before running this script.
"""
from pathlib import Path
import contextlib, io, json, os, re
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.matplotlib'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
REPORT = BASE / 'report/project2_full_report.tex'
IMAGES = BASE / 'report/overleaf_images'
KEYS = ['auto_mpg', 'concrete', 'airfoil']
TITLES = dict(zip(KEYS, ['Auto MPG', 'Concrete', 'Airfoil']))
UNITS = dict(zip(KEYS, ['MPG', 'MPa', 'dB']))


def row(*cells):
    return ' & '.join(map(str, cells)) + r' \\' + '\n'


def table(caption, label, columns, header, rows):
    return (r'\begin{table}[H]' + '\n' + r'\centering\small' + '\n'
            + r'\caption{' + caption + '}\n' + r'\label{' + label + '}\n'
            + r'\begin{tabular}{' + columns + '}\n' + r'\toprule' + '\n'
            + row(*header) + r'\midrule' + '\n' + ''.join(row(*r) for r in rows)
            + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}' + '\n\n')


def figure(path, caption, label):
    return (r'\begin{figure}[H]' + '\n' + r'\centering' + '\n'
            + r'\includegraphics[width=\textwidth]{' + path + '}\n'
            + r'\caption{' + caption + '}\n' + r'\label{' + label + '}\n'
            + r'\end{figure}' + '\n\n')


def term_tex(name, symbols):
    if name == 'one': return '1'
    if name == 'x2^1.0x0^-1.0x4^-1.5': return r'c/(f d^{3/2})'
    if name.startswith('log1p('): return r'\log(1+' + symbols[name[6:-1]] + ')'
    if '^' in name:
        original, power = name.rsplit('^', 1)
        symbol = symbols[original]
        if float(power) == .5: return r'\sqrt{' + symbol + '}'
        return symbol + '^{' + str(int(float(power))) + '}'
    if name in symbols: return symbols[name]
    parts = []
    rest = name
    for original in sorted(symbols, key=len, reverse=True):
        if rest == original or rest.startswith(original + '_'):
            parts.append(symbols[original])
            rest = rest[len(original):].lstrip('_')
            break
    if not parts: raise ValueError(name)
    if rest: parts.append(term_tex(rest, symbols))
    return r'\,'.join(parts)


def number(v):
    s = f'{v:.11g}'
    if 'e' in s:
        a, b = s.split('e')
        return a + r'\times10^{' + str(int(b)) + '}'
    return s


def symbolic_section():
    metrics = {}
    for key in KEYS:
        m = pd.read_csv(BASE / f'results/symbolic/{key}_metrics.csv')
        metrics[key] = {r.model + '_' + r.split: r for r in m.itertuples()}
    rows = []
    for key in KEYS:
        tr, te = metrics[key]['selected_train'], metrics[key]['selected_test']
        rows.append([TITLES[key], int(tr.rows), int(te.rows), int(te.terms),
                     f'{tr.r2:.4f}', f'{te.r2:.4f}', f'{te.rmse:.4f}', f'{te.mae:.4f}'])
    text = r'''\subsection{ScalaTion}

ScalaTion's \texttt{SymbolicRegression} builds candidate basis functions, and forward selection retains a subset for a linear combination of nonlinear terms. Predictor names are read from the CSV headers, so cylinders and displacement are identified in the repository's actual order. Each dataset is shuffled with Scala seed 42 and split \emph{before} any fitting or selection. The training/test sizes are 314/78 for Auto MPG, 824/206 for Concrete, and 1203/300 for Airfoil. Feature construction uses predefined operators; means, scales, numerical-rank screening, and feature selection use training rows only. The test rows are reserved until the final selected model is fitted.

Auto MPG uses powers $-2,-1,1/2,2$; Concrete uses $1,1/2,2$; Airfoil uses $1,1/2,2$, $\log(1+x)$, two-way and three-way products of distinct predictors, and $c/(f d^{3/2})$. Here $c$, $f$, and $d$ denote chord length, frequency, and suction displacement thickness. Powers and operator sets are fixed before evaluation. All transformations are finite for the observed predictors.

Each nonconstant basis function is centered and scaled using its training mean and population standard deviation. In candidate order, two-pass Gram--Schmidt screening rejects columns whose residual norm is at most $10^{-6}$ of their original standardized norm. This removes numerically redundant functions using only the training design. There are 28/24/41 initial nonconstant candidates for Auto MPG/Concrete/Airfoil and 26/24/40 after screening. Forward selection minimizes the library's training $\mathrm{sMAPE}_{IC}$ criterion over the complete selection path. The final subset is refitted by QR least squares on all outer training rows; test predictions use those same training scaling constants. No random validation calls or test responses enter selection.

Earlier randomized validation after feature selection did not reserve independent test rows. The results below replace those scores with the corrected split-first evaluation. They describe predictive performance on one fixed split. Conventional significance claims from the earlier post-validation summaries are omitted: selecting correlated nonlinear features does not establish confirmatory effects or causal relationships.

'''
    text += table('Corrected ScalaTion symbolic regression: training fit and untouched test-set performance. RMSE/MAE use each response\'s original units.',
                  'tab:scalation-symbolic-results', 'lrrrrrrr',
                  ['Dataset', r'$n_{tr}$', r'$n_{te}$', 'Terms', r'Train $R^2$', r'Test $R^2$', 'RMSE', 'MAE'], rows)
    base_rows = []
    for key in KEYS:
        b, t = metrics[key]['linear_baseline_test'], metrics[key]['selected_test']
        gain = 100 * (1 - t.rmse / b.rmse)
        base_rows.append([TITLES[key], f'{b.r2:.4f}', f'{b.rmse:.4f}', f'{t.rmse:.4f}', f'{gain:.1f}'])
    text += table('Linear baseline and symbolic regression on identical Scala test rows. Positive reductions indicate lower symbolic RMSE.',
                  'tab:symbolic-baseline', 'lrrrr',
                  ['Dataset', r'Linear $R^2$', 'Linear RMSE', 'Symbolic RMSE', r'RMSE reduction (\%)'], base_rows)
    interpretations = {
        'auto_mpg': 'The selected expression includes inverse weight, nonlinear model year, inverse displacement and horsepower, acceleration, origin, and cylinder terms. Individual signs belong to a joint nonlinear combination and should not be interpreted as isolated effects. Numeric origin coding follows the other ScalaTion models.',
        'concrete': 'The selected expression includes square-root age, linear age, material powers, and square-root material terms. The opposing age functions allow curvature in the fitted response. Terms involving water and superplasticizer occur together with correlated material variables, so coefficient signs alone do not determine monotone or causal effects.',
        'airfoil': 'The selected expression combines frequency, chord length, angle of attack, velocity, and displacement thickness through powers and interaction terms. The custom function is explicitly $c/(f d^{3/2})$. Training-only screening removes a redundant velocity-squared term, avoiding the earlier enormous cancelling coefficients. The remaining correlated terms still require joint interpretation.'}
    symbol_maps = {
        'auto_mpg': {'cylinders':'C', 'displacement':'D', 'horsepower':'H', 'weight':'W', 'acceleration':'A', 'modelyear':'Y', 'origin':'O'},
        'concrete': {'cement':'C', 'blast_furnace_slag':'S', 'fly_ash':'F', 'water':'W', 'superplasticizer':'P', 'coarse_aggregate':'G', 'fine_aggregate':'Q', 'age':'A'},
        'airfoil': {'frequency_hz':'f', 'angle_attack_deg':'a', 'chord_length_m':'c', 'free_stream_velocity_ms':'v', 'suction_displacement_thickness_m':'d'}}
    definitions = {
        'auto_mpg': r'$C,D,H,W,A,Y,O$ are cylinders, displacement, horsepower, weight, acceleration, model year (70--82), and numeric origin, respectively.',
        'concrete': r'$C,S,F,W,P,G,Q,A$ are cement, slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age, respectively.',
        'airfoil': r'$f,a,c,v,d$ are frequency (Hz), attack angle (degrees), chord length (m), free-stream velocity (m/s), and suction displacement thickness (m), respectively.'}
    for key in KEYS:
        tr, te, b = (metrics[key][n] for n in ['selected_train', 'selected_test', 'linear_baseline_test'])
        gain = 100 * (1 - te.rmse / b.rmse)
        coef = pd.read_csv(BASE / f'results/symbolic/{key}_coefficients.csv')
        k = len(coef) - 1
        text += r'\subsubsection{' + TITLES[key] + '}\n\n'
        text += (f'The model retained {k} nonconstant terms plus an intercept. Test $R^2={te.r2:.4f}$, '
                 f'RMSE $={te.rmse:.4f}$~{UNITS[key]}, and MAE $={te.mae:.4f}$~{UNITS[key]}. '
                 f'Against linear regression on the same training/test rows, RMSE falls from {b.rmse:.4f} '
                 f'to {te.rmse:.4f}~{UNITS[key]} ({gain:.1f}\\%). ' + interpretations[key] + '\n\n')
        text += definitions[key] + '\n'
        text += (r'The complete fitted expression is' + '\n' + r'\begin{equation}' + '\n'
                 + r'\widehat{y}= ' + number(coef.iloc[0].coefficient)
                 + r' + \sum_{j=1}^{' + str(k) + r'} b_j\frac{\phi_j-\mu_j}{s_j}.'
                 + '\n' + r'\end{equation}' + '\n'
                 + r'The following table specifies \emph{every} term and its fitted coefficient and training scaling constants. Values are rounded to 11 significant digits; full double-precision constants and predictions are exported in the dataset-specific \texttt{project2/results/symbolic/} CSVs. The intercept above is not centered or scaled.' + '\n\n')
        text += r'{\scriptsize\setlength{\tabcolsep}{3pt}' + '\n' + r'\begin{longtable}{rlrrr}' + '\n'
        text += r'\caption{Complete ' + TITLES[key] + r' ScalaTion symbolic expression parameters.}\label{tab:' + key.replace('_','-') + '-symbolic-complete}' + r'\\' + '\n'
        text += r'\toprule' + '\n' + row('$j$', r'$\phi_j$', '$b_j$', r'$\mu_j$', '$s_j$') + r'\midrule\endfirsthead' + '\n'
        text += r'\toprule' + '\n' + row('$j$', r'$\phi_j$', '$b_j$', r'$\mu_j$', '$s_j$') + r'\midrule\endhead' + '\n'
        for j, c in enumerate(coef.iloc[1:].itertuples(), 1):
            text += row(j, '$' + term_tex(c.term, symbol_maps[key]) + '$',
                        '$'+number(c.coefficient)+'$', '$'+number(c.training_mean)+'$', '$'+number(c.training_scale)+'$')
        text += r'\bottomrule\end{longtable}}' + '\n\n'
    # Regenerate plots exclusively from the new untouched-test predictions.
    fig, axes = plt.subplots(1,3,figsize=(15,4.5))
    for ax,key in zip(axes,KEYS):
        pred = pd.read_csv(BASE / f'results/symbolic/{key}_test_predictions.csv')
        ax.scatter(pred.y,pred.prediction,s=14,alpha=.55)
        lo=min(pred.y.min(),pred.prediction.min()); hi=max(pred.y.max(),pred.prediction.max())
        ax.plot([lo,hi],[lo,hi],'k--',lw=1)
        ax.set(title=TITLES[key],xlabel=f'Actual ({UNITS[key]})',ylabel=f'Predicted ({UNITS[key]})')
        ax.grid(alpha=.25)
    fig.tight_layout()
    path = IMAGES/'Symbolic Regression Images/scalation_symbolic_test_predictions.png'
    fig.savefig(path,dpi=180);plt.close(fig)
    text += figure('Symbolic Regression Images/scalation_symbolic_test_predictions.png',
                   'Corrected ScalaTion symbolic predictions on test rows reserved before fitting and selection. The diagonal represents perfect prediction.', 'fig:scalation-symbolic-test')
    return text,metrics


def notebook_results():
    n=json.loads((BASE/'Project2_Regular_Regression.ipynb').read_text())
    env={}
    previous=Path.cwd()
    try:
        os.chdir(BASE)
        with contextlib.redirect_stdout(io.StringIO()):
            for c in n['cells']:
                s=''.join(c.get('source',[]))
                if c['cell_type']=='code' and not s.lstrip().startswith('%'):
                    exec(compile(s,'regularized_notebook','exec'),env)
    finally: os.chdir(previous)
    return env


def regularized_section():
    env=notebook_results()
    sm=pd.read_csv(BASE/'results/regularized/auto_mpg_metrics.csv').set_index('method')
    co=pd.read_csv(BASE/'results/regularized/auto_mpg_coefficients.csv')
    cv=pd.read_csv(BASE/'results/regularized/auto_mpg_tuning.csv')
    predictions=pd.read_csv(BASE/'results/regularized/auto_mpg_test_predictions.csv')
    text=r'''\section{Regularized Regression}

Ridge and Lasso were fitted to Auto MPG in both ScalaTion and \texttt{statsmodels}, giving four required cases. Ridge penalizes squared slope coefficients; Lasso penalizes their absolute values and can yield sparse models. Both implementations standardize predictors using training statistics and leave the intercept unpenalized. Origin is retained as a numeric predictor, matching the multiple regression setup.

\subsection{statsmodels}

The notebook uses a scikit-learn seed-42 80/20 split (313 training, 79 test rows). Five-fold CV with shuffled seed-42 folds considers $\alpha\in\{0.001,0.01,0.1,1,10,100\}$. A new scaler is fitted inside each fold, and the intercept has zero penalty weight. The tuning score is mean fold RMSE. Both methods select $\alpha=0.01$. Final scalers and coefficients are fitted using the complete training set, then evaluated once on test data.

\subsection{ScalaTion}

The corrected implementation uses Scala's seed-42 shuffle (314 training, 78 test rows). For both methods, five contiguous folds of the shuffled training rows evaluate the same grid $\lambda=0.1\,2^i$, $i=0,\ldots,19$. Each fold estimates predictor means and sample standard deviations, and centers the response, using only its fitting rows. Validation predictions add that fold's response mean back. Lambda minimizes pooled CV RMSE, $\sqrt{\sum e_i^2/n_{train}}$. The final model uses complete-training statistics and adds the training response mean back, so the intercept is unpenalized. This replaces preprocessing before internal CV and the earlier penalized Scala Lasso intercept.

'''
    rows=[]
    for method in ['ridge','lasso']:
        rows.append(['statsmodels',method.title(),f'$\\alpha={env[f"best_{method}_alpha"]:g}$',f'{env[f"{method}_rmse"]:.4f}',f'{env[f"{method}_r2"]:.4f}'])
        m=sm.loc[method]
        rows.append(['ScalaTion',method.title(),f'$\\lambda={m["lambda"]:g}$',f'{m.test_rmse:.4f}',f'{m.test_r2:.4f}'])
    text+=table('Auto MPG Ridge and Lasso test-set results. The two tools use different test rows.', 'tab:regularized-results','lllrr',
                ['Tool','Method','Penalty', 'Test RMSE',r'Test $R^2$'],rows)
    names=['intercept','cylinders','displacement','horsepower','weight','acceleration','modelyear','origin']
    coeff_rows=[]
    for j,name in enumerate(names):
        values=[]
        for method in ['ridge','lasso']:
            values.append(float(np.asarray(env[f'{method}_model'].params)[j]))
        for method in ['ridge','lasso']:
            values.append(float(co[(co.method==method)&(co.term==name)].coefficient.iloc[0]))
        coeff_rows.append([name.replace('modelyear','model year')]+[f'{v:.6f}' for v in values])
    text+=table('Coefficients on standardized predictors; intercepts are in MPG. Each tool uses its own training scaler.',
                'tab:regularized-coefficients','lrrrr',['Term','SM Ridge','SM Lasso','Scala Ridge','Scala Lasso'],coeff_rows)
    text+=r'''Weight has a large negative coefficient and model year a positive coefficient in all four fitted models. statsmodels Lasso sets acceleration exactly to zero. The corrected Scala Lasso keeps a small nonzero acceleration coefficient; sparsity depends on the chosen penalty and solver tolerance. Because these are coefficients on standardized, correlated predictors, their signs describe conditional fitted associations rather than independent causal effects.

\subsection{Evaluation and Interpretation}

'''
    ols_rmse=float(np.sqrt(np.mean((np.asarray(env['y_test'])-np.asarray(env['X_test_scaled'])@np.linalg.lstsq(np.asarray(env['X_train_scaled']),np.asarray(env['y_train']),rcond=None)[0])**2)))
    text+=f'On the statsmodels test split, unregularized OLS has RMSE {ols_rmse:.4f}~MPG; on the ScalaTion split it has RMSE 3.5276~MPG. '
    text+=r'''The regularized fits do not improve test RMSE over these matched-split baselines. Regularization shrinks coefficients and can reduce variance in other samples, but this experiment does not demonstrate reduced overfitting. Alpha and lambda values cannot be compared numerically across implementations because their penalty normalizations differ. Differences between tools also reflect different test rows, scaler conventions, solvers, and tuning objectives. The two tuning scores are mean fold RMSE (statsmodels) and pooled RMSE (ScalaTion); neither is a test-set score.

'''
    # Updated tuning, coefficient, and fitted-output visual evidence for all four cases.
    folder=IMAGES/'Regular Regression Images'
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for method in ['ridge','lasso']:
        axes[0].semilogx(env['alphas'],env[f'{method}_cv_scores'],'o-',label=method.title())
        d=cv[cv.method==method]
        axes[1].semilogx(d['lambda'],d.cv_rmse,'o-',label=method.title())
    axes[0].set(title='statsmodels: fold mean RMSE',xlabel='Alpha',ylabel='CV RMSE (MPG)')
    axes[1].set(title='ScalaTion: pooled fold RMSE',xlabel='Lambda',ylabel='CV RMSE (MPG)')
    for a in axes:a.legend();a.grid(alpha=.25)
    fig.tight_layout();fig.savefig(folder/'tuning_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4.5))
    x=np.arange(7); width=.19
    for q,(tool,method) in enumerate([('statsmodels','ridge'),('statsmodels','lasso'),('ScalaTion','ridge'),('ScalaTion','lasso')]):
        vals=np.asarray(env[f'{method}_model'].params)[1:] if tool=='statsmodels' else co[(co.method==method)&(co.term!='intercept')].coefficient.to_numpy()
        ax.bar(x+(q-1.5)*width,vals,width,label=f'{tool} {method.title()}')
    ax.set_xticks(x,['cylinders','displacement','horsepower','weight','acceleration','model year','origin']);ax.set_ylabel('Standardized slope');ax.axhline(0,color='k',lw=.6);ax.legend(ncol=2);fig.tight_layout();fig.savefig(folder/'coefficient_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,(tool,method) in zip(axes.flat,[('statsmodels','ridge'),('statsmodels','lasso'),('ScalaTion','ridge'),('ScalaTion','lasso')]):
        if tool=='statsmodels':
            y=np.asarray(env['y_test']);yp=np.asarray(env[f'{method}_model'].predict(env['X_test_scaled']))
        else:
            d=predictions[predictions.method==method];y=d.y.to_numpy();yp=d.prediction.to_numpy()
        ax.scatter(y,yp,s=20,alpha=.6);lo=min(y.min(),yp.min());hi=max(y.max(),yp.max());ax.plot([lo,hi],[lo,hi],'k--');ax.set(title=f'{tool} {method.title()}',xlabel='Actual MPG',ylabel='Predicted MPG');ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(folder/'test_actual_vs_predicted.png',dpi=180);plt.close(fig)
    text+=figure('Regular Regression Images/tuning_comparison.png','Training CV tuning curves for Ridge and Lasso. Each candidate in the corrected ScalaTion curves uses the same folds and fold-specific preprocessing.','fig:reg-tuning')
    text+=figure('Regular Regression Images/coefficient_comparison.png','Ridge/Lasso slope coefficients on standardized predictors for all four Auto MPG models.','fig:reg-coef')
    text+=figure('Regular Regression Images/test_actual_vs_predicted.png','Observed versus predicted test MPG. Each ScalaTion panel uses the corrected coefficients and untouched outer test rows; statsmodels uses its own split.','fig:reg-test')
    return text


def comparison(metrics):
    values={'auto_mpg':(.8519,2.7495),'concrete':(.8357,6.5062),'airfoil':(.8159,3.0372)}
    rows=[]
    for key in KEYS:
        t=metrics[key]['selected_test']
        rows+=[[TITLES[key],'ScalaTion',f'{t.r2:.4f}',f'{t.rmse:.4f}'],['','PySR',f'{values[key][0]:.4f}',f'{values[key][1]:.4f}']]
    text=r'\subsection{Comparison of ScalaTion and PySR}'+'\n\n'
    text+=table('Symbolic regression on each tool\'s own test split. ScalaTion scores use the corrected split-first procedure.','tab:symbolic-comparison','llrr',['Dataset','Method',r'Test $R^2$','RMSE'],rows)
    text+=r'''Both methods provide explicit nonlinear expressions, but their search strategies and evaluation splits differ. ScalaTion builds a predefined nonlinear basis and selects terms from it, whereas PySR searches expression structures directly. ScalaTion uses Scala seed-42 shuffling and PySR uses scikit-learn's seed-42 split, so their test rows differ. These scores describe each experiment; they do not establish that one tool is better on a common test population.

Auto MPG has the largest ScalaTion test $R^2$ among its three datasets, and PySR gives a compact two-variable equation using weight and model year. For Concrete and Airfoil, PySR has lower reported test RMSE than the corrected ScalaTion model on its own split. Within ScalaTion, all three selected symbolic models improve RMSE over linear regression evaluated on identical rows. The complete ScalaTion equations above expose every selected term, coefficient, and scaling constant; individual nonlinear coefficients require joint interpretation.

'''
    return text


def forward_figures(source):
    """Use the report's corrected candidate-size metrics, not the library's full DF."""
    for key, label in [('auto_mpg','fs-auto-path'),('concrete','fs-concrete-path')]:
        block = source.split(r'\label{tab:' + label + '}')[1].split(r'\end{tabular}')[0]
        records = []
        for line in block.splitlines():
            parts = line.split('&')
            if len(parts) != 6 or not parts[0].strip().isdigit():
                continue
            numbers = []
            for cell in parts[2:]:
                match = re.search(r'-?\d+(?:\.\d+)?', cell)
                numbers.append(float(match.group()))
            records.append([int(parts[0]), *numbers])
        values = np.asarray(records)
        assert len(values) == (8 if key == 'auto_mpg' else 9)
        fig,ax=plt.subplots(figsize=(8,5))
        for j,name,color in [(1,r'$R^2$','red'),(2,r'Adjusted $R^2$','green'),(3,'sMAPE','blue'),(4,r'$R^2_{cv}$','black')]:
            ax.plot(values[:,0],values[:,j],'o-',color=color,label=name)
        ax.set(xlabel='Number of predictors (excluding intercept)',ylabel='Percent',title=TITLES[key]+': forward selection')
        ax.legend();ax.grid(alpha=.25);fig.tight_layout()
        fig.savefig(IMAGES/f'Forward Selection Images/fs_{key}_r2_vs_n.png',dpi=180)
        plt.close(fig)


def polish_report(source):
    """Keep the submission focused on the final experiment and identify evidence."""
    source = source.replace(
        "This replaces preprocessing before internal CV and the earlier penalized Scala Lasso intercept.", "")
    old = ("Earlier randomized validation after feature selection did not reserve independent test rows. "
           "The results below replace those scores with the corrected split-first evaluation. "
           "They describe predictive performance on one fixed split. Conventional significance claims "
           "from the earlier post-validation summaries are omitted: selecting correlated nonlinear features "
           "does not establish confirmatory effects or causal relationships.")
    source = source.replace(old, "These results describe predictive performance on one fixed split. "
                            "Selecting correlated nonlinear features does not establish confirmatory effects "
                            "or causal relationships.")
    source = source.replace(
        "Training-only screening removes a redundant velocity-squared term, avoiding the earlier enormous "
        "cancelling coefficients. The remaining correlated terms still require joint interpretation.",
        "Training-only screening removes a redundant velocity-squared term. The $d$ and $\\log(1+d)$ "
        "terms have large opposing standardized coefficients because their basis functions are nearly "
        "collinear, so their contributions must be read together. Individual coefficients remain sensitive "
        "to this correlation.")
    source = source.replace(
        r"Repository input paths are resolved to the data directory, and \texttt{modelyear} is mapped to "
        r"\texttt{model\_year} for the Auto MPG notebook. Displayed equation constants are rounded; small "
        r"differences when reevaluating them in double precision do not replace the saved prediction metrics.", "")
    source = re.sub(r'\bCorrected\s+', '', source)
    source = re.sub(r'\bcorrected\s+', '', source)
    source = source.replace("The implementation uses Scala's", "The ScalaTion implementation uses Scala's")
    source = re.sub(r'\bStatsmodels\b', lambda _: r'\texttt{statsmodels}', source)
    # One equation number for each multiline expression.
    source = source.replace(r'\begin{align}', r'\begin{equation}' + '\n' + r'\begin{aligned}')
    source = source.replace(r'\end{align}', r'\end{aligned}' + '\n' + r'\end{equation}')
    explanation = ("Ridge penalization discourages large slope estimates among correlated cylinders, "
                   "displacement, horsepower, and weight, allowing their shared information to remain in "
                   "the fitted model. Lasso can remove a redundant predictor, as with acceleration in "
                   r"\texttt{statsmodels}; a zero conditional coefficient does not establish that the "
                   "variable has no marginal association with MPG.")
    marker = "Because these are coefficients on standardized, correlated predictors, their signs describe conditional fitted associations rather than independent causal effects."
    if explanation not in source:
        source = source.replace(marker, marker + ' ' + explanation)
    # These six screenshots are Python OLS summaries, not ScalaTion screenshots.
    start = source.index(r'\subsection{ScalaTion}', source.index(r'\section{Multiple Regression}'))
    end = source.index(r'\subsection{\texttt{statsmodels}}', start)
    scala = source[start:end]
    figures = re.findall(r'\\begin\{figure\}\[H\].*?\\end\{figure\}', scala, flags=re.S)
    if figures:
        assert len(figures) == 6
        for figure_block in figures:
            scala = scala.replace(figure_block, '')
        source = source[:start] + scala + source[end:]
        figures = [re.sub(r'(\\caption\{[^}]*?)\.\}', lambda m: m[1] + r' (\texttt{statsmodels} output).}', f)
                   for f in figures]
        before = source.index(r'\subsection{Results Comparison}', start)
        source = source[:before] + '\n\n'.join(figures) + '\n\n' + source[before:]
    return '\n'.join(line.rstrip() for line in source.splitlines()) + '\n'


def main():
    s=REPORT.read_text()
    s=s.replace(r'\usepackage{threeparttable}',r'\usepackage{threeparttable}'+'\n'+r'\usepackage{longtable}') if r'\usepackage{longtable}' not in s else s
    start=s.index(r'\section{Regularized Regression}');end=s.index(r'\section{Forward Feature Selection}')
    s=s[:start]+regularized_section()+s[end:]
    start=s.index(r'\section{Forward Feature Selection}');end=s.index(r'\section{Transformed Regression}')
    fs=s[start:end].replace('314 rows for Auto MPG and 824 for Concrete','313 rows for Auto MPG and 823 for Concrete').replace('(314 rows)','(313 rows)').replace('(824 rows)','(823 rows)')
    fs=fs.replace("Each candidate model was fitted on ScalaTion's default 80\\% training split", "At steps 1 onward, candidate models were fitted on ScalaTion's default training range")
    fs=fs.replace('Every candidate model was also evaluated with the default 5-fold cross-validation', 'The intercept-only step is fitted on the full dataset. Each candidate subset was also evaluated with the default 5-fold cross-validation on the full dataset')
    fs=fs.replace('which is the more honest estimate of out-of-sample performance', 'which evaluates the fixed selected subset across folds; selection was not repeated inside each fold, so this is not a nested-CV estimate of the complete selection procedure')
    fs=fs.replace('so dropping acceleration costs nothing','so dropping acceleration changes the descriptive fit very little')
    fs=fs.replace('so selection stops at six of seven predictors','so the retained subset contains six of seven predictors even though the path visits all seven')
    s=s[:start]+fs+s[end:]
    forward_figures(s)
    symbolic,metrics=symbolic_section()
    start=s.index(r'\subsection{ScalaTion}',s.index(r'\section{Symbolic Regression}'));end=s.index(r'\subsection{PySR}',start)
    s=s[:start]+symbolic+s[end:]
    start=s.index(r'\subsection{Comparison of ScalaTion and PySR}')
    s=s[:start]+comparison(metrics)+r'\end{document}'+'\n'
    # Repository provenance and exact assignment allocation belong in the report.
    extra=r'''
The cleaned UCI data contain 392 rows and 8 numeric columns for Auto MPG (target MPG), 1,030 rows and 9 columns for Concrete (target compressive strength in MPa), and 1,503 rows and 6 columns for Airfoil (target scaled sound pressure in dB). Auto MPG's 398-row numeric source has six missing-horsepower rows; removing those rows and excluding car names yields the 392-row modeling file. Direct comparison with the supplied UCI archives confirms the cleaned values and row order (Concrete differs only at floating-point roundoff). The report covers 12 ScalaTion cases, 5 statsmodels cases, and 3 PySR cases: 20 total. Log, square root, and Box--Cox are applied to the response for each of the two transformed-regression datasets.
'''
    if 'Direct comparison with the supplied UCI archives' not in s:
        i=s.index(r'\section{Multiple Regression}');s=s[:i]+extra+'\n'+s[i:]
    # Explain precision/name mapping without altering the historical PySR search outputs.
    intro=r'''The reported PySR scores are from the saved search outputs. Auto MPG used serial deterministic search with seed 42; Concrete and Airfoil used multithreading, so their search trajectories are not guaranteed repeatable from a seed alone. Repository input paths are resolved to the data directory, and \texttt{modelyear} is mapped to \texttt{model\_year} for the Auto MPG notebook. Displayed equation constants are rounded; small differences when reevaluating them in double precision do not replace the saved prediction metrics.

'''
    marker='The operator set included the binary operators'
    if 'search trajectories are not guaranteed' not in s:s=s.replace(marker,intro+marker)
    s=s.replace(r'\date{September 2026}',r'\date{October 2026}')
    REPORT.write_text(polish_report(s))
    print('Updated', REPORT)


if __name__=='__main__':main()
