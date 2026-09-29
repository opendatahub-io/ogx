# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from ogx.cli.stack.utils import print_subcommand_description
from ogx.cli.subcommand import Subcommand

from .praxis.cmd import PraxisMigrate


class MigrateParser(Subcommand):
    """The ``ogx migrate`` command group — one-time data migrations (OGX -> Praxis).

    Mirrors ``StackParser``: registers a ``migrate`` subparser whose sub-commands
    are the individual migrations. The whole group is excisable — deleting
    ``src/ogx/cli/migrate/`` and the one registration line in ``ogx.py`` fully
    removes it once migrations are complete.
    """

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "migrate",
            prog="ogx migrate",
            description="One-time data migrations for OGX deployments.",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self.parser.set_defaults(func=lambda args: self.parser.print_help())

        subparsers = self.parser.add_subparsers(title="migrate_subcommands")
        PraxisMigrate.create(subparsers)
        print_subcommand_description(self.parser, subparsers)
