import typer

from microtcn.cli import comp, export, plot, speed, test, train

app = typer.Typer(
    name="microtcn",
    help="Efficient neural networks for real-time modeling of analog dynamic range compression.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("train")(train.train)
app.command("train-all")(train.train_all)
app.command("test")(test.test)
app.command("comp")(comp.comp)
app.command("export")(export.export)
app.command("speed")(speed.speed)
app.add_typer(plot.app, name="plot", help="Generate plots.")


if __name__ == "__main__":
    app()
