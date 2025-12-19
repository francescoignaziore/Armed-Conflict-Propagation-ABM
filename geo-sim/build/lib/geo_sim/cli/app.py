import warnings

warnings.filterwarnings("ignore")
import typer

app = typer.Typer(
    no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False
)


from .tiff_alignment import tiff_alignment

app.command("tiff-alignment")(tiff_alignment)

from .roads import roads_to_length

app.command("roads-to-tiff")(roads_to_length)

from .buildings import buildings_feats

app.command("buildings-to-tiff")(buildings_feats)


from .natural import natural_features

app.command("natural-to-tiff")(natural_features)

from .water import waters_feats

app.command("waters-to-tiff")(waters_feats)

from .landuse import landuse_feats

app.command("landuse-to-tiff")(landuse_feats)


from .sim import run_simulation

app.command("run-simulation")(run_simulation)


def main():
    app(standalone_mode=True)


if __name__ == "__main__":
    main()
