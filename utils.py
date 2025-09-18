import matplotlib.pyplot as plt
import numpy as np

def plot_grid(Field):
    # --- plot ---
    plt.figure(figsize=(8,6))
    plt.imshow(Field, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(f"Mag")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def plot_ang_grid(Field, r_in, r_ext, ang_1 = 0, ang_2 = 120):

    n_r, n_a = Field.shape  # infere pelas dimensões
    ang_1 = np.deg2rad(ang_1)
    ang_2 = np.deg2rad(ang_2)

    r_edges = np.linspace(r_in, r_ext, n_r+1)
    a_edges = np.linspace(ang_1, ang_2, n_a+1)
    R, A = np.meshgrid(r_edges, a_edges, indexing="ij")

    X = R * np.cos(A)
    Y = R * np.sin(A)

    fig, ax = plt.subplots(subplot_kw={"aspect":"equal"})
    pcm = ax.pcolormesh(X, Y, Field, cmap="viridis", shading="auto")

    cbar = fig.colorbar(pcm, ax=ax)
    ax.set_title("Magnitude")
    plt.show()