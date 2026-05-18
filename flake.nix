{
  description = "BTC 15-minute price direction predictor";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python313;
        pythonPkgs = python.pkgs;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pythonPkgs.pandas
            pythonPkgs.scikit-learn
            pythonPkgs.joblib
            pythonPkgs.pytest
          ];

          shellHook = ''
            echo "BTC Price Predictor"
            echo "  prepare:  python -m src.prepare_data"
            echo "  train:    python -m src.train"
            echo "  evaluate: python -m src.evaluate"
            echo "  predict:  python -m src.predict --help"
            echo "  test:     pytest tests/ -v"
          '';
        };
      }
    );
}
