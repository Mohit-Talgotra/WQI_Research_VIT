import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
R=[("GAM (Direct WQI)",14.82,-0.196,"#fedd8e"),("ParameterCatBoost Default\n(parameter-first)",15.19,-0.257,"#91c4de"),("ParameterCatBoost Tuned\n(parameter-first)",15.24,-0.265,"#b6dbee"),("Direct CatBoost",15.31,-0.277,"#fdae61"),("Persistence (Naive Lag-1)",16.45,-0.475,"#1b7837"),("Seasonal Mean Baseline",16.45,-0.475,"#bdbdbd"),("Location Mean Baseline",19.70,-1.115,"#969696"),("Hierarchical State-Space\n(Partial Pooling)",20.78,-1.353,"#d73027")]
plt.rcParams["font.family"]="serif"
fig,ax=plt.subplots(figsize=(10,5.5))
for i,(n,r,r2,c) in enumerate(R):
    ax.barh(i,r,color=c,edgecolor="black",height=0.62); ax.text(r+0.3,i,f"RMSE: {r:.2f}  (R²: {r2:.3f})",va="center",fontsize=10)
ax.set_yticks(range(len(R))); ax.set_yticklabels([x[0] for x in R]); ax.invert_yaxis()
ax.axvline(16.45,color="#1b7837",ls="--",lw=2,label="Persistence baseline (RMSE = 16.45); paired-bootstrap\nΔRMSE intervals all include 0 (Table 9)")
ax.set_xlim(0,30); ax.set_xlabel("RMSE on 2025 holdout, causal Kalman fill (lower is better)")
ax.set_title("Forecasting Benchmark Comparison, Leakage-Corrected\n(2025 Temporal Holdout, N = 63 Real Observed Months)")
ax.grid(axis="x",ls=":",alpha=.5); ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.13),fontsize=9.5,framealpha=.95)
fig.tight_layout(); fig.savefig(Path(__file__).resolve().parents[2]/"figures/fig7_corrected_leakage_fix.png",dpi=300)
