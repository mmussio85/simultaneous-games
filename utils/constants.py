import matplotlib.pyplot as plt

COLORS = {
    "FP":     "#2196F3",  # azul
    "RM":     "#4CAF50",  # verde
    "IQL":    "#FF9800",  # naranja
    "JAL-AM": "#9C27B0",  # violeta
    "JALAM": "#9C27B0",  # violeta
    "Random": "#E91E63",  # rosado
}

JAL_AM_PALETTE = [
    COLORS["JAL-AM"],  # Purple 500
    "#EA80FC",         # Purple A100
    "#EC407A",         # Pink 400
]



plt.rcParams.update({
    "figure.dpi":      150,
    "figure.figsize":  (9, 5),
    "font.size":       12,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "axes.grid":       True,
    "grid.alpha":      0.3,
})

N_EPISODES = 10_000
N_SEEDS    = 10

IQL_KWARGS = dict(
    alpha=0.1,
    gamma=0.0,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
)

JALAM_KWARGS = dict(
    alpha=0.1,
    gamma=0.0,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
)

IQL_FORAGING_KWARGS = dict(
    alpha=0.1,
    gamma=0.9,
    epsilon=1.0,
    epsilon_decay=0.999,
    epsilon_min=0.05,
)

JALAM_FORAGING_KWARGS = dict(
    alpha=0.1,
    gamma=0.9,
    epsilon=1.0,
    epsilon_decay=0.999,
    epsilon_min=0.05,
)
