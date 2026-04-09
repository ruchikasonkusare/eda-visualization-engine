# from eda.file_loader import load_file
# from eda.attribut_type import attribute_types
# from eda.visualization import generate_eda_report       
# from eda.nulls import handle_missing_values
# from eda.outliers import treat_outliers

# def run_eda(file_path=None):
#     df, file_path = load_file(file_path)
#     print(f"Loaded file: {file_path}")
    
#     col_types = attribute_types(df)
#     print("Column Types:")
#     print(col_types)
    
#     null_summary = handle_missing_values(df)
#     print("Missing Values Summary:")
#     print(null_summary)
    
#     df = treat_outliers(df)
#     print("Outliers treated.")
    
#     generate_eda_report(df, col_types)
#     print("EDA report generated: output/eda_report.pdf")
    
# if __name__ == "__main__":
#     run_eda()

"""
Data Visualization Insight Engine
==================================
Automatically detects column types, cleans data, detects outliers,
removes duplicates, normalizes data, and generates all suitable charts
with explanations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f172a',
    'axes.facecolor':   '#1e293b',
    'axes.edgecolor':   '#334155',
    'axes.labelcolor':  '#e2e8f0',
    'xtick.color':      '#94a3b8',
    'ytick.color':      '#94a3b8',
    'text.color':       '#e2e8f0',
    'grid.color':       '#334155',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.titlesize':   12,
    'axes.titleweight': 'bold',
    'axes.titlepad':    12,
    'legend.facecolor': '#1e293b',
    'legend.edgecolor': '#334155',
})

PALETTE   = ['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ef4444',
             '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#84cc16']
ACCENT    = '#6366f1'
GOOD      = '#10b981'
WARN      = '#f59e0b'
DANGER    = '#ef4444'


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════

class DataInspector:
    """Analyses every column and returns a rich type profile."""

    def __init__(self, df: pd.DataFrame):
        self.df      = df.copy()
        self.profile = {}
        self._inspect()

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _is_datetime(series: pd.Series) -> bool:
        try:
            pd.to_datetime(series.dropna().head(20), infer_datetime_format=True)
            return True
        except Exception:
            return False

    @staticmethod
    def _is_binary(series: pd.Series) -> bool:
        return series.dropna().nunique() == 2

    @staticmethod
    def _is_ordinal(series: pd.Series) -> bool:
        ordinal_keywords = ['low', 'medium', 'high', 'poor', 'good', 'excellent',
                            'never', 'rarely', 'sometimes', 'often', 'always',
                            'small', 'large', 'very']
        vals = [str(v).lower() for v in series.dropna().unique()]
        return any(kw in vals for kw in ordinal_keywords)

    # ── main inspection ───────────────────────────────────────────────────────
    def _inspect(self):
        df = self.df
        for col in df.columns:
            s        = df[col]
            n_total  = len(s)
            n_null   = s.isna().sum()
            n_unique = s.nunique()
            sample   = s.dropna()

            info = {
                'n_total':  n_total,
                'n_null':   int(n_null),
                'null_pct': round(n_null / n_total * 100, 2),
                'n_unique': int(n_unique),
                'dtype':    str(s.dtype),
            }

            # ── numeric ──────────────────────────────────────────────────────
            if pd.api.types.is_numeric_dtype(s):
                is_discrete = (s.dropna() == s.dropna().astype(int)).all() and n_unique < 20
                skew        = float(sample.skew())
                kurt        = float(sample.kurt())
                q1, q3      = sample.quantile(0.25), sample.quantile(0.75)
                iqr         = q3 - q1
                n_outliers  = int(((sample < q1 - 1.5 * iqr) | (sample > q3 + 1.5 * iqr)).sum())

                info.update({
                    'kind':       'discrete_numeric' if is_discrete else 'continuous_numeric',
                    'min':        float(sample.min()),
                    'max':        float(sample.max()),
                    'mean':       float(sample.mean()),
                    'median':     float(sample.median()),
                    'std':        float(sample.std()),
                    'skew':       skew,
                    'kurt':       kurt,
                    'n_outliers': n_outliers,
                    'binary':     self._is_binary(s),
                })

            # ── datetime ─────────────────────────────────────────────────────
            elif self._is_datetime(s):
                info['kind'] = 'datetime'

            # ── categorical ──────────────────────────────────────────────────
            else:
                info.update({
                    'kind':    'ordinal'     if self._is_ordinal(s)
                               else 'binary' if self._is_binary(s)
                               else 'categorical',
                    'top':     sample.value_counts().head(5).to_dict(),
                    'binary':  self._is_binary(s),
                })

            self.profile[col] = info

    def numeric_cols(self):
        return [c for c, v in self.profile.items()
                if v['kind'] in ('continuous_numeric', 'discrete_numeric')]

    def categorical_cols(self):
        return [c for c, v in self.profile.items()
                if v['kind'] in ('categorical', 'binary', 'ordinal')]

    def datetime_cols(self):
        return [c for c, v in self.profile.items() if v['kind'] == 'datetime']


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA CLEANER
# ══════════════════════════════════════════════════════════════════════════════

class DataCleaner:
    """Removes duplicates, fills/drops nulls, caps outliers, normalises."""

    def __init__(self, df: pd.DataFrame, inspector: DataInspector):
        self.original   = df.copy()
        self.df         = df.copy()
        self.inspector  = inspector
        self.report     = {}
        self._clean()

    def _clean(self):
        df = self.df

        # 1. Duplicates
        n_dup = df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        self.report['duplicates_removed'] = int(n_dup)

        # 2. Missing values
        null_report = {}
        for col in df.columns:
            n_null = df[col].isna().sum()
            if n_null == 0:
                continue
            pct = n_null / len(df)
            if pct > 0.5:                                   # >50% missing → drop col
                df.drop(columns=[col], inplace=True)
                null_report[col] = 'column dropped (>50% missing)'
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
                null_report[col] = f'{n_null} filled with median'
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                null_report[col] = f'{n_null} filled with mode'
        self.report['missing'] = null_report

        # 3. Outlier capping (IQR ×1.5)
        num_cols     = self.inspector.numeric_cols()
        num_cols     = [c for c in num_cols if c in df.columns]
        outlier_rep  = {}
        for col in num_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr    = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out  = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_out:
                df[col] = df[col].clip(lo, hi)
                outlier_rep[col] = f'{n_out} outliers capped'
        self.report['outliers'] = outlier_rep

        # 4. Normalisation (Min-Max → [0,1])
        norm_rep = {}
        for col in num_cols:
            if col not in df.columns:
                continue
            rng = df[col].max() - df[col].min()
            if rng > 0:
                df[col + '_norm'] = ((df[col] - df[col].min()) / rng).round(4)
                norm_rep[col] = col + '_norm'
        self.report['normalized'] = norm_rep

        self.df = df


# ══════════════════════════════════════════════════════════════════════════════
# 3. RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class RecommendationEngine:
    """Rule-based engine that maps data conditions to chart types."""

    def __init__(self, df: pd.DataFrame, inspector: DataInspector):
        self.df        = df
        self.inspector = inspector
        self.charts    = []   # list of dicts {type, cols, reason, priority}

    def recommend(self):
        num  = self.inspector.numeric_cols()
        cat  = self.inspector.categorical_cols()
        dt   = self.inspector.datetime_cols()
        prof = self.inspector.profile

        # ── single numeric ────────────────────────────────────────────────────
        for col in num:
            skew = prof[col].get('skew', 0)
            skew_label = ('right-skewed' if skew > 1
                          else 'left-skewed' if skew < -1
                          else 'approximately normal')
            self._add('histogram',   [col], priority=1,
                      reason=f'Shows distribution of {col}. Data is {skew_label} (skew={skew:.2f}).')
            self._add('boxplot_single', [col], priority=2,
                      reason=f'Reveals spread, median and outliers of {col}.')
            self._add('violin',      [col], priority=3,
                      reason=f'Violin combines box-plot and KDE for {col}.')
            self._add('kde',         [col], priority=2,
                      reason=f'Smooth density curve for {col}.')

        # ── single categorical ────────────────────────────────────────────────
        for col in cat:
            n_unique = prof[col]['n_unique']
            self._add('bar',  [col], priority=1,
                      reason=f'Frequency comparison across {n_unique} categories in {col}.')
            if n_unique <= 8:
                self._add('pie', [col], priority=2,
                          reason=f'Proportion view of {col} ({n_unique} slices).')
                self._add('donut', [col], priority=3,
                          reason=f'Donut chart for {col} composition.')

        # ── numeric × categorical ─────────────────────────────────────────────
        for n in num:
            for c in cat:
                if prof[c]['n_unique'] <= 15:
                    self._add('boxplot_group', [n, c], priority=1,
                              reason=f'Spread of {n} across {c} categories.')
                    self._add('violin_group',  [n, c], priority=2,
                              reason=f'Distribution shape of {n} per {c}.')
                    self._add('bar_group',     [n, c], priority=3,
                              reason=f'Mean {n} per {c} as grouped bars.')

        # ── two numeric ───────────────────────────────────────────────────────
        for i, n1 in enumerate(num):
            for n2 in num[i+1:]:
                corr = self.df[[n1, n2]].corr().iloc[0, 1]
                label = ('strong positive' if corr > 0.7
                         else 'strong negative' if corr < -0.7
                         else 'moderate' if abs(corr) > 0.4
                         else 'weak')
                self._add('scatter', [n1, n2], priority=1,
                          reason=f'{label} correlation (r={corr:.2f}) between {n1} and {n2}.')
                self._add('hexbin',  [n1, n2], priority=3,
                          reason=f'Hexbin density for {n1} vs {n2} (avoids overplotting).')

        # ── correlation heatmap ───────────────────────────────────────────────
        if len(num) >= 3:
            self._add('heatmap', num, priority=1,
                      reason=f'Correlation matrix across {len(num)} numeric variables.')

        # ── time series ───────────────────────────────────────────────────────
        for d in dt:
            for n in num:
                self._add('line', [d, n], priority=1,
                          reason=f'Trend of {n} over time ({d}).')
                self._add('area', [d, n], priority=2,
                          reason=f'Area chart emphasises cumulative trend of {n}.')

        # ── multi-numeric ─────────────────────────────────────────────────────
        if len(num) >= 2:
            self._add('pairplot', num[:5], priority=2,
                      reason=f'Pairwise relationships among {min(len(num),5)} numeric columns.')

        if len(num) >= 3:
            self._add('parallel', num[:6], priority=3,
                      reason='Parallel coordinates reveal multivariate patterns.')

        return sorted(self.charts, key=lambda x: x['priority'])

    def _add(self, chart_type, cols, priority, reason):
        self.charts.append({
            'type':     chart_type,
            'cols':     cols,
            'priority': priority,
            'reason':   reason,
        })


# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class PlotGenerator:
    """Draws every recommended chart and saves to a folder."""

    def __init__(self, df: pd.DataFrame, output_dir: str = 'charts'):
        self.df  = df
        self.out = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, fig, name):
        path = os.path.join(self.out, name + '.png')
        fig.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    def _fig(self, w=10, h=6):
        return plt.subplots(figsize=(w, h))

    # ── individual chart methods ──────────────────────────────────────────────

    def histogram(self, col):
        fig, ax = self._fig()
        data = self.df[col].dropna()
        ax.hist(data, bins='auto', color=ACCENT, edgecolor='#0f172a', alpha=0.85)
        ax.axvline(data.mean(),   color=GOOD,  ls='--', lw=1.5, label=f'Mean {data.mean():.2f}')
        ax.axvline(data.median(), color=WARN,  ls=':',  lw=1.5, label=f'Median {data.median():.2f}')
        ax.set(title=f'Distribution of {col}', xlabel=col, ylabel='Frequency')
        ax.legend(); ax.grid(True, axis='y')
        return self._save(fig, f'histogram_{col}')

    def kde(self, col):
        fig, ax = self._fig()
        data = self.df[col].dropna()
        data.plot.kde(ax=ax, color=ACCENT, lw=2)
        ax.fill_between(ax.lines[0].get_xdata(), ax.lines[0].get_ydata(),
                        alpha=0.2, color=ACCENT)
        ax.set(title=f'Density of {col}', xlabel=col, ylabel='Density')
        ax.grid(True)
        return self._save(fig, f'kde_{col}')

    def boxplot_single(self, col):
        fig, ax = self._fig(8, 5)
        data = self.df[col].dropna()
        bp = ax.boxplot(data, vert=False, patch_artist=True,
                        boxprops=dict(facecolor=ACCENT, alpha=0.7),
                        medianprops=dict(color=GOOD, lw=2),
                        whiskerprops=dict(color='#94a3b8'),
                        capprops=dict(color='#94a3b8'),
                        flierprops=dict(marker='o', color=DANGER, alpha=0.5))
        ax.set(title=f'Box Plot — {col}', xlabel=col)
        ax.grid(True, axis='x')
        return self._save(fig, f'boxplot_{col}')

    def violin(self, col):
        fig, ax = self._fig(8, 5)
        sns.violinplot(y=self.df[col].dropna(), ax=ax,
                       color=ACCENT, inner='box', alpha=0.8)
        ax.set(title=f'Violin Plot — {col}', ylabel=col)
        ax.grid(True)
        return self._save(fig, f'violin_{col}')

    def bar(self, col):
        fig, ax = self._fig()
        counts = self.df[col].value_counts().head(15)
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]
        counts.plot.bar(ax=ax, color=colors, edgecolor='#0f172a', width=0.7)
        ax.set(title=f'Frequency — {col}', xlabel=col, ylabel='Count')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y')
        for p in ax.patches:
            ax.annotate(str(int(p.get_height())),
                        (p.get_x() + p.get_width()/2, p.get_height()),
                        ha='center', va='bottom', fontsize=8, color='#e2e8f0')
        return self._save(fig, f'bar_{col}')

    def pie(self, col):
        fig, ax = self._fig(8, 8)
        counts = self.df[col].value_counts().head(8)
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
               colors=PALETTE[:len(counts)], startangle=140,
               wedgeprops=dict(edgecolor='#0f172a', linewidth=1.5))
        ax.set_title(f'Composition — {col}')
        return self._save(fig, f'pie_{col}')

    def donut(self, col):
        fig, ax = self._fig(8, 8)
        counts = self.df[col].value_counts().head(8)
        wedges, texts, autotexts = ax.pie(
            counts, labels=counts.index, autopct='%1.1f%%',
            colors=PALETTE[:len(counts)], startangle=140,
            wedgeprops=dict(width=0.5, edgecolor='#0f172a', linewidth=1.5))
        ax.set_title(f'Donut — {col}')
        return self._save(fig, f'donut_{col}')

    def boxplot_group(self, num_col, cat_col):
        fig, ax = self._fig(12, 6)
        cats = self.df[cat_col].value_counts().head(10).index
        data = [self.df[self.df[cat_col] == c][num_col].dropna() for c in cats]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color=GOOD, lw=2),
                        whiskerprops=dict(color='#94a3b8'),
                        capprops=dict(color='#94a3b8'),
                        flierprops=dict(marker='o', color=DANGER, alpha=0.4))
        for patch, color in zip(bp['boxes'], PALETTE):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set_xticklabels(cats, rotation=30)
        ax.set(title=f'{num_col} by {cat_col}', xlabel=cat_col, ylabel=num_col)
        ax.grid(True, axis='y')
        return self._save(fig, f'boxplot_{num_col}_by_{cat_col}')

    def violin_group(self, num_col, cat_col):
        fig, ax = self._fig(12, 6)
        subset = self.df[[num_col, cat_col]].dropna()
        top_cats = subset[cat_col].value_counts().head(8).index
        subset = subset[subset[cat_col].isin(top_cats)]
        sns.violinplot(x=cat_col, y=num_col, data=subset, ax=ax,
                       palette=PALETTE[:len(top_cats)], inner='box', alpha=0.8)
        ax.set(title=f'{num_col} distribution by {cat_col}')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y')
        return self._save(fig, f'violin_{num_col}_by_{cat_col}')

    def bar_group(self, num_col, cat_col):
        fig, ax = self._fig(12, 6)
        means = self.df.groupby(cat_col)[num_col].mean().nlargest(12)
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(means))]
        means.plot.bar(ax=ax, color=colors, edgecolor='#0f172a', width=0.7)
        ax.set(title=f'Mean {num_col} by {cat_col}', xlabel=cat_col, ylabel=f'Mean {num_col}')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y')
        return self._save(fig, f'bargroup_{num_col}_by_{cat_col}')

    def scatter(self, col1, col2):
        fig, ax = self._fig()
        x, y = self.df[col1].dropna(), self.df[col2].dropna()
        min_len = min(len(x), len(y))
        x, y = x.iloc[:min_len], y.iloc[:min_len]
        ax.scatter(x, y, alpha=0.5, color=ACCENT, s=20, edgecolors='none')
        m, b, r, p, _ = stats.linregress(x, y)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, m * xline + b, color=DANGER, lw=2,
                label=f'r = {r:.2f}')
        ax.set(title=f'{col1} vs {col2}', xlabel=col1, ylabel=col2)
        ax.legend(); ax.grid(True)
        return self._save(fig, f'scatter_{col1}_vs_{col2}')

    def hexbin(self, col1, col2):
        fig, ax = self._fig()
        x = self.df[col1].dropna()
        y = self.df[col2].dropna()
        min_len = min(len(x), len(y))
        hb = ax.hexbin(x.iloc[:min_len], y.iloc[:min_len],
                       gridsize=25, cmap='YlOrRd', mincnt=1)
        plt.colorbar(hb, ax=ax, label='count')
        ax.set(title=f'Hexbin — {col1} vs {col2}', xlabel=col1, ylabel=col2)
        return self._save(fig, f'hexbin_{col1}_vs_{col2}')

    def heatmap(self, cols):
        fig, ax = self._fig(max(8, len(cols)), max(6, len(cols)))
        corr = self.df[cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, ax=ax,
                    linewidths=0.5, linecolor='#0f172a',
                    annot_kws={'size': 9})
        ax.set_title('Correlation Heatmap')
        return self._save(fig, 'heatmap_correlation')

    def line(self, time_col, num_col):
        fig, ax = self._fig(14, 5)
        df_sorted = self.df[[time_col, num_col]].dropna().sort_values(time_col)
        try:
            df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
        except Exception:
            pass
        ax.plot(df_sorted[time_col], df_sorted[num_col],
                color=ACCENT, lw=1.8, marker='o', markersize=3, alpha=0.85)
        ax.set(title=f'{num_col} over {time_col}', xlabel=time_col, ylabel=num_col)
        ax.grid(True); plt.xticks(rotation=30)
        return self._save(fig, f'line_{num_col}_over_{time_col}')

    def area(self, time_col, num_col):
        fig, ax = self._fig(14, 5)
        df_sorted = self.df[[time_col, num_col]].dropna().sort_values(time_col)
        try:
            df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
        except Exception:
            pass
        ax.fill_between(df_sorted[time_col], df_sorted[num_col],
                        alpha=0.4, color=ACCENT)
        ax.plot(df_sorted[time_col], df_sorted[num_col],
                color=ACCENT, lw=2)
        ax.set(title=f'Area — {num_col} over {time_col}', xlabel=time_col, ylabel=num_col)
        ax.grid(True); plt.xticks(rotation=30)
        return self._save(fig, f'area_{num_col}_over_{time_col}')

    def pairplot(self, cols):
        df_sub = self.df[cols].dropna()
        g = sns.pairplot(df_sub, diag_kind='kde', plot_kws={'alpha': 0.4, 'color': ACCENT},
                         diag_kws={'color': ACCENT})
        g.fig.suptitle('Pair Plot', y=1.02, color='#e2e8f0')
        g.fig.set_facecolor('#0f172a')
        path = os.path.join(self.out, 'pairplot.png')
        g.fig.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0f172a')
        plt.close('all')
        return path

    def parallel(self, cols):
        from pandas.plotting import parallel_coordinates
        df_sub = self.df[cols].dropna().head(500)
        # create dummy class for coloring
        df_sub = df_sub.copy()
        df_sub['_grp'] = pd.qcut(df_sub[cols[0]], q=4, labels=['Q1','Q2','Q3','Q4'],
                                  duplicates='drop')
        fig, ax = self._fig(14, 6)
        parallel_coordinates(df_sub, '_grp', ax=ax, color=PALETTE[:4], alpha=0.4)
        ax.set_title('Parallel Coordinates')
        ax.legend(title=cols[0])
        ax.grid(True)
        return self._save(fig, 'parallel_coordinates')

    # ── dispatch ──────────────────────────────────────────────────────────────
    def draw(self, chart):
        t    = chart['type']
        cols = chart['cols']
        try:
            if   t == 'histogram':      return self.histogram(cols[0])
            elif t == 'kde':            return self.kde(cols[0])
            elif t == 'boxplot_single': return self.boxplot_single(cols[0])
            elif t == 'violin':         return self.violin(cols[0])
            elif t == 'bar':            return self.bar(cols[0])
            elif t == 'pie':            return self.pie(cols[0])
            elif t == 'donut':          return self.donut(cols[0])
            elif t == 'boxplot_group':  return self.boxplot_group(cols[0], cols[1])
            elif t == 'violin_group':   return self.violin_group(cols[0], cols[1])
            elif t == 'bar_group':      return self.bar_group(cols[0], cols[1])
            elif t == 'scatter':        return self.scatter(cols[0], cols[1])
            elif t == 'hexbin':         return self.hexbin(cols[0], cols[1])
            elif t == 'heatmap':        return self.heatmap(cols)
            elif t == 'line':           return self.line(cols[0], cols[1])
            elif t == 'area':           return self.area(cols[0], cols[1])
            elif t == 'pairplot':       return self.pairplot(cols)
            elif t == 'parallel':       return self.parallel(cols)
        except Exception as e:
            return None   # skip silently


# ══════════════════════════════════════════════════════════════════════════════
# 5. REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Produces a self-contained HTML report with all charts embedded."""

    def __init__(self, df_original, df_clean, inspector, cleaner,
                 charts_meta, output_dir):
        self.df_orig   = df_original
        self.df_clean  = df_clean
        self.inspector = inspector
        self.cleaner   = cleaner
        self.charts    = charts_meta   # list of {type, cols, reason, path}
        self.out       = output_dir

    def _img_tag(self, path):
        import base64
        if not path or not os.path.exists(path):
            return ''
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:12px;">'

    def generate(self):
        prof   = self.inspector.profile
        report = self.cleaner.report
        rows   = len(self.df_orig)
        cols   = len(self.df_orig.columns)

        # ── column profile table ──────────────────────────────────────────────
        col_rows = ''
        for col, info in prof.items():
            kind      = info['kind']
            null_pct  = info['null_pct']
            n_unique  = info['n_unique']
            extra     = ''
            if 'mean' in info:
                extra = f"mean={info['mean']:.2f}, skew={info['skew']:.2f}, outliers={info['n_outliers']}"
            elif 'top' in info:
                top = list(info['top'].keys())[:3]
                extra = 'top: ' + ', '.join(str(t) for t in top)
            badge_color = {
                'continuous_numeric': '#6366f1',
                'discrete_numeric':   '#8b5cf6',
                'categorical':        '#06b6d4',
                'binary':             '#10b981',
                'ordinal':            '#f59e0b',
                'datetime':           '#ec4899',
            }.get(kind, '#94a3b8')
            col_rows += f'''
            <tr>
              <td><code>{col}</code></td>
              <td><span style="background:{badge_color};padding:2px 8px;border-radius:20px;font-size:11px;color:#fff">{kind}</span></td>
              <td>{null_pct}%</td>
              <td>{n_unique}</td>
              <td style="font-size:12px;color:#94a3b8">{extra}</td>
            </tr>'''

        # ── cleaning report ───────────────────────────────────────────────────
        clean_items = f"<li>Duplicates removed: <strong>{report.get('duplicates_removed', 0)}</strong></li>"
        for col, msg in report.get('missing', {}).items():
            clean_items += f'<li>Missing — <code>{col}</code>: {msg}</li>'
        for col, msg in report.get('outliers', {}).items():
            clean_items += f'<li>Outliers — <code>{col}</code>: {msg}</li>'
        for col, norm in report.get('normalized', {}).items():
            clean_items += f'<li>Normalised — <code>{col}</code> → <code>{norm}</code></li>'

        # ── chart cards ──────────────────────────────────────────────────────
        chart_cards = ''
        for c in self.charts:
            if not c.get('path'):
                continue
            img      = self._img_tag(c['path'])
            col_str  = ' + '.join(c['cols'])
            chart_cards += f'''
            <div class="chart-card">
              <div class="chart-meta">
                <span class="chart-type">{c["type"].replace("_"," ").title()}</span>
                <span class="chart-cols">{col_str}</span>
              </div>
              {img}
              <p class="chart-reason">💡 {c["reason"]}</p>
            </div>'''

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data Visualization Insight Engine</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:"Segoe UI",Arial,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}}
  header{{background:linear-gradient(135deg,#4f46e5,#06b6d4);padding:40px 32px;text-align:center}}
  header h1{{font-size:2rem;font-weight:800;letter-spacing:-0.5px}}
  header p{{color:#bae6fd;margin-top:8px}}
  .container{{max-width:1400px;margin:0 auto;padding:32px 24px}}
  .section{{background:#1e293b;border:1px solid #334155;border-radius:16px;padding:28px;margin-bottom:28px}}
  .section h2{{font-size:1.1rem;font-weight:700;color:#818cf8;margin-bottom:16px;display:flex;align-items:center;gap:8px}}
  .stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:0}}
  .stat-card{{background:#0f172a;border-radius:12px;padding:20px;text-align:center;border:1px solid #334155}}
  .stat-card .num{{font-size:2rem;font-weight:800;color:#6366f1}}
  .stat-card .lbl{{font-size:12px;color:#64748b;margin-top:4px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{background:#0f172a;color:#94a3b8;padding:10px 14px;text-align:left;font-weight:600;border-bottom:1px solid #334155}}
  td{{padding:10px 14px;border-bottom:1px solid #1e293b;vertical-align:middle}}
  tr:hover td{{background:#0f172a}}
  code{{background:#0f172a;padding:2px 6px;border-radius:4px;color:#818cf8;font-size:12px}}
  .clean-list{{list-style:none;display:flex;flex-direction:column;gap:8px}}
  .clean-list li{{background:#0f172a;padding:10px 14px;border-radius:8px;font-size:13px;border-left:3px solid #6366f1}}
  .charts-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:24px}}
  .chart-card{{background:#0f172a;border:1px solid #334155;border-radius:16px;padding:20px}}
  .chart-meta{{display:flex;align-items:center;gap:10px;margin-bottom:14px}}
  .chart-type{{background:#6366f1;color:#fff;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700}}
  .chart-cols{{font-size:12px;color:#64748b}}
  .chart-reason{{margin-top:12px;font-size:12px;color:#94a3b8;line-height:1.5}}
  footer{{text-align:center;padding:24px;color:#334155;font-size:12px}}
</style>
</head>
<body>
<header>
  <h1>📊 Data Visualization Insight Engine</h1>
  <p>Automatic analysis · Cleaning · Outlier detection · Smart chart recommendations</p>
  <p style="margin-top:6px;font-size:12px;color:#bae6fd">Generated {datetime.now().strftime("%d %b %Y %H:%M")}</p>
</header>

<div class="container">

  <!-- Overview -->
  <div class="section">
    <h2>📋 Dataset Overview</h2>
    <div class="stats-grid">
      <div class="stat-card"><div class="num">{rows}</div><div class="lbl">Rows</div></div>
      <div class="stat-card"><div class="num">{cols}</div><div class="lbl">Columns</div></div>
      <div class="stat-card"><div class="num">{len(self.inspector.numeric_cols())}</div><div class="lbl">Numeric</div></div>
      <div class="stat-card"><div class="num">{len(self.inspector.categorical_cols())}</div><div class="lbl">Categorical</div></div>
      <div class="stat-card"><div class="num">{len(self.inspector.datetime_cols())}</div><div class="lbl">Datetime</div></div>
      <div class="stat-card"><div class="num">{report.get("duplicates_removed",0)}</div><div class="lbl">Duplicates Removed</div></div>
    </div>
  </div>

  <!-- Column Profile -->
  <div class="section">
    <h2>🔬 Column Profile (Auto-detected Types)</h2>
    <table>
      <thead><tr><th>Column</th><th>Detected Type</th><th>Missing %</th><th>Unique</th><th>Stats / Top Values</th></tr></thead>
      <tbody>{col_rows}</tbody>
    </table>
  </div>

  <!-- Cleaning Report -->
  <div class="section">
    <h2>🧹 Auto-Cleaning Report</h2>
    <ul class="clean-list">{clean_items}</ul>
  </div>

  <!-- Charts -->
  <div class="section">
    <h2>📈 Generated Visualizations ({len([c for c in self.charts if c.get("path")])} charts)</h2>
    <div class="charts-grid">{chart_cards}</div>
  </div>

</div>
<footer>Data Visualization Insight Engine · Auto-generated report</footer>
</body>
</html>'''

        path = os.path.join(self.out, 'report.html')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        return path


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class VisualizationEngine:
    """One-call entry point."""

    def __init__(self, filepath: str, output_dir: str = 'visualization_output'):
        self.filepath   = filepath
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        print(f"\n{'═'*55}")
        print("  📊  Data Visualization Insight Engine")
        print(f"{'═'*55}\n")

        # 1. Load
        print("⏳ Loading dataset …")
        ext = os.path.splitext(self.filepath)[1].lower()
        if ext in ('.xls', '.xlsx'):
            df = pd.read_excel(self.filepath)
        else:
            df = pd.read_csv(self.filepath)
        print(f"   ✅ {len(df)} rows × {len(df.columns)} columns loaded\n")

        # 2. Inspect
        print("🔬 Detecting column types …")
        inspector = DataInspector(df)
        for col, info in inspector.profile.items():
            print(f"   {col:<25} → {info['kind']}")

        # 3. Clean
        print("\n🧹 Cleaning data …")
        cleaner = DataCleaner(df, inspector)
        r = cleaner.report
        print(f"   Duplicates removed : {r['duplicates_removed']}")
        print(f"   Columns with missing: {len(r['missing'])}")
        print(f"   Columns with outliers: {len(r['outliers'])}")
        print(f"   Columns normalised : {len(r['normalized'])}")
        df_clean = cleaner.df

        # Re-inspect clean df (norm cols added)
        inspector2 = DataInspector(df_clean)

        # 4. Recommend
        print("\n💡 Recommending charts …")
        engine = RecommendationEngine(df_clean, inspector)
        charts = engine.recommend()
        print(f"   {len(charts)} chart recommendations generated")

        # 5. Draw (limit to avoid 100s of files; keep top 40)
        charts_dir = os.path.join(self.output_dir, 'charts')
        generator  = PlotGenerator(df_clean, charts_dir)
        print("\n🎨 Generating plots …")
        drawn = 0
        for c in charts[:40]:
            path = generator.draw(c)
            c['path'] = path
            if path:
                drawn += 1
                print(f"   ✅ {c['type']:<20} {' + '.join(c['cols'])}")
        print(f"\n   {drawn} charts saved to {charts_dir}/")

        # 6. Report
        print("\n📄 Building HTML report …")
        reporter = ReportGenerator(df, df_clean, inspector, cleaner,
                                   charts, self.output_dir)
        report_path = reporter.generate()
        print(f"   ✅ Report saved → {report_path}")
        print(f"\n{'═'*55}\n  Done! Open report.html in your browser.\n{'═'*55}\n")
        return report_path


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys
    print(sys)
    if len(sys.argv) < 2:
        print("Usage: python engine.py data/linkedin_data.csv")
        sys.exit(1)
    VisualizationEngine(sys.argv[1]).run()