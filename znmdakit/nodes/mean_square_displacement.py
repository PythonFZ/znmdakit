import MDAnalysis.analysis.msd as msd
import numpy as np
import pandas as pd
import zntrack
from MDAnalysis import Universe
from scipy.stats import linregress
from pathlib import Path
import matplotlib.pyplot as plt
import pint

from znmdakit.transformations import UnWrap, get_com_transform

ureg = pint.UnitRegistry()


class EinsteinMSD(zntrack.Node):
    universe: Universe = zntrack.deps()
    select: str = zntrack.params()
    results: pd.DataFrame = zntrack.plots(y="msd", x="lagtimes")
    apply_com_transform: bool = zntrack.params(False)

    # TODO: these two should be read from the h5md file
    timestep: float = zntrack.params(None)
    sampling_rate: int = zntrack.params(None)

    def run(self) -> None:
        universe = self.universe

        transformations = []

        if self.apply_com_transform:
            transformations.extend(get_com_transform(universe))
        universe.trajectory.add_transformations(
            *transformations, UnWrap()
        )  # UnWrap is the last one!

        MSD = msd.EinsteinMSD(universe, select=self.select, msd_type="xyz", fft=True)
        MSD.run(verbose=True)

        if self.timestep is not None and self.sampling_rate is not None:
            dt = self.timestep * self.sampling_rate
        else:
            dt = universe.trajectory.dt

        lagtimes = np.arange(MSD.n_frames) * dt

        self.results = pd.DataFrame(
            {
                "lagtimes": lagtimes,
                "msd": MSD.results.timeseries,
            }
        )
        # set the index to lagtimes
        self.results.set_index("lagtimes", inplace=True)


class SelfDiffusionFromMSD(zntrack.Node):
    """Compute the self-diffusion coefficient from the mean square displacement."""

    data: EinsteinMSD = zntrack.deps()

    start_time: float = zntrack.params(0)
    end_time: float = zntrack.params(-1)

    fit_figure: Path = zntrack.plots_path(zntrack.nwd / "msd_fit.png")

    metrics: dict = zntrack.metrics()

    def run(self):
        if self.data.timestep is None or self.data.sampling_rate is None:
            raise ValueError("timestep/sampling_rate must be set")

        # How is the msd one shorter than the lagtimes?
        # TODO: use loc on index and then iloc on the values
        linear_model = linregress(
            self.data.results.index[self.start_time : self.end_time],
            self.data.results.msd.iloc[self.start_time : self.end_time],
        )

        diff = (linear_model.slope  / 6) * ureg.angstrom**2 / ureg.picosecond
        # diff = diff.to(ureg.meter**2 / ureg.second)
        diff = diff.to(ureg.angstrom**2 / ureg.nanosecond)
        # convert to nm / s

        self.metrics = {
            "slope": linear_model.slope,
            "intercept": linear_model.intercept,
            "rvalue": linear_model.rvalue,
            "diffusion": diff.magnitude,
        }

        self.fit_figure.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)  # Adjust figure size and resolution

        # Plot MSD and fit line with improved styling
        ax.plot(self.data.results.index, self.data.results.msd, label="MSD", lw=2, color="royalblue")
        ax.plot(
            self.data.results.index,
            linear_model.slope * self.data.results.index + linear_model.intercept,
            label="Fit",
            lw=2,
            linestyle="--",
            color="darkorange",
        )

        # Labels with larger fonts
        ax.set_xlabel("Time t / ps", fontsize=12)
        ax.set_ylabel("MSD / Å²", fontsize=12)

        # Add vertical lines with transparency
        ax.axvline(self.start_time, color="crimson", linestyle="--", lw=1.5, alpha=0.75)
        ax.axvline(self.end_time, color="crimson", linestyle="--", lw=1.5, alpha=0.75)

        # Add a grid for readability
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Add diffusion coefficient and R² value in a less intrusive way
        ax.text(
            0.05,  # Move slightly left
            0.85,  # Move slightly up
            f"Self-diffusion: {diff.magnitude:.2f} Å²/ns\nR²: {linear_model.rvalue:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray", boxstyle="round,pad=0.3"),
        )

        fig.savefig(self.fit_figure, bbox_inches="tight")
