import pandas as pd
import zntrack
from MDAnalysis import Universe
from MDAnalysis.analysis import rdf
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from znmdakit.transformations import get_com_transform


class InterRDF(zntrack.Node):
    universe: Universe = zntrack.deps()
    g1: str = zntrack.params()
    g2: str = zntrack.params()
    nbins: int = zntrack.params()
    range: str | tuple = zntrack.params("auto")

    apply_com_transform: bool = zntrack.params(
        False
    )  # replace the position of the first atom in each residue with the center of mass and name it COM

    results: pd.DataFrame = zntrack.plots(y="g(r)", x="r")
    figure_path: Path = zntrack.plots_path(zntrack.nwd / "rdf.png")

    def run(self):
        universe = self.universe

        if self.apply_com_transform:
            transformations = get_com_transform(universe)
            universe.trajectory.add_transformations(*transformations)

        RDF = rdf.InterRDF(
            universe.select_atoms(self.g1),
            universe.select_atoms(self.g2),
            nbins=1000,
            range=(0, universe.dimensions[0] / 2)
            if self.range == "auto"
            else self.range,
        )
        RDF.run(verbose=True)

        self.results = pd.DataFrame(
            {"r": RDF.results.bins[1:], "g(r)": RDF.results.rdf[1:]}
        )

        self.results.set_index("r", inplace=True)
        fig = self.get_figure()
        fig.savefig(self.figure_path, bbox_inches="tight")

    def get_figure(self) -> plt.Figure:
        sns.set_theme(style="whitegrid")  # Use a clean seaborn style

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)  # Set figure size and resolution
        ax.plot(self.results.index, self.results["g(r)"], color="tab:blue", lw=2)

        ax.set_ylabel("Radial Distribution Function, g(r)", fontsize=12)
        ax.set_xlabel("Distance, r (Å)", fontsize=12)
        ax.set_title(f"InterRDF: {self.g1} - {self.g2}", fontsize=14, fontweight="bold")

        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)

        return fig
